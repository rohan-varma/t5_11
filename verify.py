# main hq file for t5 training and prediction

import os
import argparse
from datasets_grammar.grammar_dataset import grammar
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# from torchvision import datasets, transforms


# for grammar correction
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# for generation
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import DataCollatorForSeq2Seq

import functools
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    CPUOffload,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    FullStateDictConfig,
    StateDictType,
)


from torch.distributed.fsdp.wrap import (
    enable_wrap,
    wrap,
)

from policies import mixed_precision

from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader
from pathlib import Path
from torch.utils.data import DataLoader

from torch.distributed._shard.checkpoint import (
    save_state_dict,
    load_state_dict,
    FileSystemReader,
    FileSystemWriter,
)

# from nlp import load_metric
# from nlp import load_dataset

from sklearn.model_selection import train_test_split
import time
from datetime import datetime

# local imports
import verify
import policies

import datasets_grammar as dg
import tqdm

# config
import config

# some globals
g_port = "12369"
g_addr = "localhost"


def _is_rank_0():
    return 0 == os.getenv("RANK")


def get_date_of_run():
    """create date and time for file save uniqueness
    example: 2022-05-07-08:31:12_PM'
    """
    date_of_run = datetime.now().strftime("%Y-%m-%d-%I:%M:%S_%p")
    print(f"--> current date and time of run = {date_of_run}")
    return date_of_run


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch fsdp T5.11 Example")
    parser.add_argument("--save-dir", default="/model_chkpt", type=str)
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=False,
        help="For Saving the current Model",
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 2022)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        metavar="N",
        help="number of epochs to train (default: 14)",
    )
    args = parser.parse_args()
    return args


# ----------------   Main functions --------------------
def get_policies(cfg, fsdp_unit_params=1000000):

    """establish current policies for mixed precision and fsdp wrapping"""

    mixed_precision_policy = None
    wrapping_policy = None

    # mixed precision -----
    if cfg.use_mixed_precision:
        bf16_ready = verify.bf16_ready

        if bf16_ready:
            mixed_precision_policy = policies.bfSixteen
            print(f"bFloat16 enabled for mixed precision - using bfSixteen policy")
        else:
            # mixed_precision_policy = policies.fpSixteen
            print(f"bFloat16 support not present. Not using for mixed precision")

    # wrapping policy -------
    # print(f"**overriding mp to fp16 - remove")
    # mixed_precision_policy = policies.fpSixteen

    wrapping_policy = policies.get_t5_wrapper(fsdp_unit_params)

    return mixed_precision_policy, wrapping_policy


def setup(rank, world_size, cfg):
    # os.environ["MASTER_ADDR"] = g_addr
    # os.environ["MASTER_PORT"] = cfg.host_port

    # initialize the process group
    dist.init_process_group("nccl")  # , rank=rank, world_size=world_size)


def setup_environ_flags(cfg):
    os.environ["TORCH_SHOW_CPP_STACKTRACES"] = str(1)
    if cfg.nccl_debug_handler:
        os.environ["NCCL_ASYNC_ERROR_HANDLING"] = str(1)


def cleanup():
    dist.destroy_process_group()


def clear_gpu_cache(rank=None):
    print(f"clearing cache for rank {rank}")
    torch.cuda.empty_cache()


def setup_tasks(rank, world_size, cfg):
    """keep the basic setup list here"""
    setup(rank, world_size, cfg)
    # clear_gpu_cache() - need to call torch set device first?
    # set_printing()
    setup_environ_flags(cfg)


# ----------  Training ----------------------------------------------------------
def train(
    args,
    model,
    local_rank,
    rank,
    world_size,
    train_loader,
    optimizer,
    epoch,
    sampler=None,
    profiler=None,
):
    model.train()
    ddp_loss = torch.zeros(2).to(local_rank)

    if sampler:
        sampler.set_epoch(epoch)
    if rank == 0:
        inner_pbar = tqdm.tqdm(
            range(len(train_loader)), colour="blue", desc="r0 Training Epoch"
        )
    i = 0
    for batch in train_loader:
        i = i + 1
        if i > 10:
            break
        for key in batch.keys():
            batch[key] = batch[key].to(local_rank)

        """print("************************")
        print(
            "train_loader",
            type(batch),
            batch["source_ids"].size(),
            batch["source_mask"].size(),
            batch["target_ids"].size(),
        )
        print("************************")
        """
        optimizer.zero_grad()
        output = model(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            labels=batch["target_ids"],
        )
        # print("##############################")
        # print(output.keys())
        # print("##############################")
        loss = output["loss"]
        loss.backward()
        optimizer.step()
        ddp_loss[0] += loss.item()
        ddp_loss[1] += len(batch)
        if rank == 0:
            inner_pbar.update(1)
        if profiler:
            profiler.step()

    dist.reduce(ddp_loss, 0, op=dist.ReduceOp.SUM)
    train_accuracy = ddp_loss[0] / ddp_loss[1]
    if rank == 0:
        inner_pbar.close()

        print(
            f"Train Epoch: \t{epoch}, Loss: \t{train_accuracy:.4f}"
        )  # .format(epoch, train_accuracy))
    return train_accuracy


# ---- Validation ---------------


def validation(model, local_rank, rank, world_size, test_loader):
    model.eval()
    correct = 0
    ddp_loss = torch.zeros(3).to(local_rank)
    if rank == 0:
        inner_pbar = tqdm.tqdm(
            range(len(test_loader)), colour="green", desc="r0 Validation Epoch"
        )
    with torch.no_grad():
        for batch in test_loader:
            for key in batch.keys():
                batch[key] = batch[key].to(local_rank)
            output = model(
                input_ids=batch["source_ids"],
                attention_mask=batch["source_mask"],
                labels=batch["target_ids"],
            )
            ddp_loss[0] += output["loss"].item()  # sum up batch loss
            ddp_loss[1] += len(batch)

            if rank == 0:
                inner_pbar.update(1)
            # pred = output.logits.argmax(
            #    dim=1, keepdim=True
            # )  # get the index of the max log-probability
            # ddp_loss[1] += pred.eq(batch["target_ids"].view_as(pred)).sum().item()
            # ddp_loss[2] += len(batch)

    dist.reduce(ddp_loss, 0, op=dist.ReduceOp.SUM)
    val_loss = ddp_loss[0] / ddp_loss[1]

    if rank == 0:
        # test_loss = ddp_loss[0] / ddp_loss[1]
        inner_pbar.close()
        print(f"Validation Loss: {val_loss:.4f}")
    return val_loss


# ---- fsdp checkpointing verification --------------------------------------
def fsdp_save_load_state_dict_no_reshard(model, model_name, save_name, wrapping_policy, mp_policy, rank, local_rank):
    device = torch.device("cuda:0")
    model_save_name = save_name + "full-state.pt"

    state_dict_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, state_dict_cfg):
        cpu_state = model.state_dict()
    print(f"rank {rank}  done w full state dict")

    if rank == 0:
        print(f"-->rank {rank} saving model full state dict...")
        torch.save(cpu_state, model_save_name)
        print(f"-->rank {rank} saved {model_save_name} to disk")

    # verify full state dict loading
    # question: although saving can only need 2x model size CPU memory, but 
    # full state dict loading will double the module size for every rank, 
    # it will be 16x model size CPU memory, can deferred module directly load state dict?
    # or only rank0 load the model and broadcast to other ranks, that only works for 
    # auto wrapping; Double check whether there is CPU memory redundancy while loading 
    # state dict.
    dist.barrier()
    print(f"-->rank {rank} loading full state dict from {model_save_name} ...")
    load_cpu_state = torch.load(model_save_name)
    print(f"-->rank {rank} loaded full state dict from {model_save_name} ...")
    new_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    new_model = FSDP(
        new_model,
        auto_wrap_policy=wrapping_policy,
        mixed_precision=mp_policy,
    ).to(local_rank)
    new_model.load_state_dict(load_cpu_state)
    print(f"rank {rank} loaded model into every rank")
    with FSDP.summon_full_params(new_model):
        with FSDP.summon_full_params(model):
            params = list(model.parameters())
            params_new = list(new_model.parameters())
            for i, j in zip(params, params_new):
                assert torch.equal(i, j)

    # verify local state dict saving and loading without resharding 
    # using torch.save()/torch.load() and torch.distributed._shard.checkpoint.save_state_dict()
    # /torch.distributed._shard.checkpoint.load_state_dict()
    
    # model_save_name = save_name + f"local-state-{rank}.pt"
    model_save_name = save_name + "local_state_dir"
    new_model1 = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    new_model1 = FSDP(
        new_model1,
        auto_wrap_policy=wrapping_policy,
        mixed_precision=mp_policy,
    ).to(local_rank)

    with FSDP.state_dict_type(
        model, StateDictType.LOCAL_STATE_DICT
    ), FSDP.state_dict_type(new_model1, StateDictType.LOCAL_STATE_DICT):
        local_state = model.state_dict()
        print(f"rank {rank}  done w local state dict")
        print(f"--> rank {rank} saving model local state dict ...")
        # torch.save(local_state, model_save_name)
        save_state_dict(
            state_dict=local_state,
            storage_writer=FileSystemWriter(model_save_name),
        )
        print(f"--> rank {rank} saved {model_save_name} to disk")
        dist.barrier()
        print(f"--> rank {rank} loading model local states from {model_save_name} ...")
        new_local_state = new_model1.state_dict()
        load_state_dict(
            state_dict=new_local_state,
            storage_reader=FileSystemReader(model_save_name),
        )
        print(f"--> rank {rank} loaded local state dict from {model_save_name} ...")
        new_model1.load_state_dict(new_local_state)
        print(f"--> rank {rank} loaded local state dict to every rank ...")

        with FSDP.summon_full_params(new_model1):
            with FSDP.summon_full_params(model):
                params = list(model.parameters())
                params_new = list(new_model1.parameters())
                for i, j in zip(params, params_new):
                    assert torch.equal(i, j)

        dist.barrier()

        params = list(model.parameters())
        params_new = list(new_model1.parameters())
        for i, j in zip(params, params_new):
            assert torch.equal(i, j)


def fsdp_load_state_dict_reshard(
    model,
    model_name, 
    save_name, 
    wrapping_policy, 
    mp_policy, 
    rank, 
    local_rank
):
    # verify local state dict saving and loading with resharding 
    # using torch.distributed._shard.checkpoint.save_state_dict()
    # / torch.distributed._shard.checkpoint.load_state_dict()
        
        
    # loading local state dicts after resharding, verify its correctness
    # loading full state dict as baseline for model
    device = torch.device("cuda:0")
    model_save_name = save_name + "full-state.pt"
    print(f"-->rank {rank} loading model full state from {model_save_name} ...")
    load_cpu_state = torch.load(model_save_name)
    model.load_state_dict(load_cpu_state)
    print(f"rank {rank} loaded model full state into every rank")

    # loading resharded local state dict for verification
    model_save_name = save_name + "local_state_dir"
    new_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    new_model = FSDP(
        new_model,
        auto_wrap_policy=wrapping_policy,
        mixed_precision=mp_policy,
    ).to(local_rank)
    print(f"--> rank {rank} loading model local states from {model_save_name} ...")
    with FSDP.state_dict_type(new_model, StateDictType.LOCAL_STATE_DICT):
        new_local_state = new_model.state_dict()
        load_state_dict(
            state_dict=new_local_state,
            storage_reader=FileSystemReader(model_save_name),
        )
        print(f"--> rank {rank} loaded local state dict from {model_save_name} ...")
        new_model.load_state_dict(new_local_state)
        print(f"--> rank {rank} loaded local state dict to model ...")

    dist.barrier()

    with FSDP.summon_full_params(new_model):
        with FSDP.summon_full_params(model):
            params = list(model.parameters())
            params_new = list(new_model.parameters())
            for i, j in zip(params, params_new):
                assert torch.equal(i, j)
    
    dist.barrier()
    
    # remove model_save_name for next run
    shutil.rmtree(model_save_name, ignore_errors=True)

# ---- fsdp main ------------------------------------------------------------


def fsdp_main(args):
    """main process within each process"""
    cfg = config.train_config()  # loads from defaults

    # torchrun specific
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if rank == 0:
        print(f"--> running with these defaults {cfg}")
        time_of_run = get_date_of_run()

    setup_tasks(rank, world_size, cfg)

    fsdp_unit_params = cfg.fsdp_unit_size
    batch_size = cfg.batch_size
    if rank == 0:
        print(f"\n BatchSize = {batch_size}\n")

    val_batch_size = cfg.val_batch_size

    mp_policy, wrapping_policy = get_policies(cfg, fsdp_unit_params)

    model_name = cfg.model_name  # "google/t5-v1_1-small"  #   #
    if rank == 0:
        print(f"--> training for model {model_name}")

    printable_model_name = str.replace(model_name, "/", "=")
    save_name = printable_model_name + "-"

    # t5-base
    # google/t5-v1_1-small
    # google/t5-v1_1-base
    # google/t5-v1_1-large
    # google/t5-v1_1-xl  #3b
    # google/t5-v1_1-xxl #11b

    # grammar correction
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer)

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # summarization
    # model = T5ForConditionalGeneration.from_pretrained(model_name)
    # tokenizer = T5Tokenizer.from_pretrained(model_name)
    # dataset_name = "jfleg_train.csv"

    if rank == 0:
        print(f"--> Training for {model_name}")
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n--> {model_name} has {total_params/1e6} Million params\n")

        # print(f"{dataset_name} contains: {dataset.keys()}")
        # print("Size of {dataset_name} train dataset: ", dataset["train"].shape)
        # print(
        #    "Size of {dataset_name} Validation dataset: ", dataset["validation"].shape
        # )

    # ____________ create batch dataset
    train_name = None
    if cfg.dataset_train:
        train_name = cfg.dataset_train

    train_dataset = dg.get_dataset(tokenizer, train_name, 512, 512, True)
    if 0 == os.getenv("RANK"):
        print(f"--> Training Set Len = {len(train_dataset)}")
        print(f"using dataset {train_name}")
    # print("bailing")

    val_dataset = dg.get_dataset(tokenizer, cfg.dataset_test, 512, 512, True)
    if 0 == os.getenv("RANK"):
        print(f"--> Validation set len = {len(val_dataset)}")
        print(f"using dataset {cfg.dataset_test}")

    sampler1 = DistributedSampler(
        train_dataset, rank=rank, num_replicas=world_size, shuffle=True
    )
    sampler2 = DistributedSampler(val_dataset, rank=rank, num_replicas=world_size)

    print(f"batch size = {batch_size}")

    train_kwargs = {"batch_size": batch_size, "sampler": sampler1}
    test_kwargs = {"batch_size": val_batch_size, "sampler": sampler2}
    cuda_kwargs = {
        "num_workers": cfg.num_workers_dataloader,
        "pin_memory": True,
        "shuffle": False,
    }
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(val_dataset, **test_kwargs)

    torch.cuda.set_device(local_rank)
    clear_gpu_cache(local_rank)

    # init_start_event = torch.cuda.Event(enable_timing=True)
    # init_end_event = torch.cuda.Event(enable_timing=True)

    # init_start_event.record()

    # model = model.to(rank)
    # model = DDP(model)
    if cfg.activation_checkpointing:
        model.gradient_checkpointing_enable()
        print(f"Activation checkpointing enabled\n")

    model = FSDP(
        model,
        auto_wrap_policy=wrapping_policy,
        mixed_precision=mp_policy,
    )
    # move model to gpu
    model.to(local_rank)

    if rank == 0 and cfg.print_sharding_plan:
        print(f"model ")
        fn = printable_model_name + "-sharded_layout.txt"
        with open(fn, "w") as external_file:
            header_text = (
                f"model = {model_name}, sharded with {fsdp_unit_params} parameters\n"
            )
            print(header_text, file=external_file)
            total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            milli_params = total_params * 4 / 1e6
            print(
                f"\n--> {model_name} has {milli_params} Million params\n",
                file=external_file,
            )
            print(f"model wrapping = \n{model}\n\n", file=external_file)

            external_file.close()

    lr = 0.0008
    gamma = 0.7
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
    epochs = cfg.num_epochs
    if rank == 0:
        print(f"Training for {epochs} epochs")

    best_train_accuracy = float("-inf")
    best_val_loss = float("inf")
    curr_val_loss = float("inf")

    # --- main training loop - todo, this needs to be modularized
    if rank == 0:
        dur = []
        train_acc_tracking = []
        val_acc_tracking = []
        training_start_time = time.time()

    torch_profiler = None
    """with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=1, warmup=2, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            "fsdp_v100/profile_traces"
        ),
        profile_memory=True,
        with_stack=False,
        record_shapes=True,
    ) as torch_profiler:
    """
    if rank == 0 and cfg.track_memory:
        fn = cfg.model_name + "memory_tracking.txt"
        mem_alloc_tracker = []
        mem_reserved_tracker = []

    for epoch in range(1, epochs + 1):
        if rank == 0:
            print(f"\n--> Starting Epoch {epoch}")

            t0 = time.time()
        train_accuracy = train(
            args,
            model,
            local_rank,
            rank,
            world_size,
            train_loader,
            optimizer,
            epoch,
            sampler=sampler1,
            profiler=torch_profiler,
        )

        if cfg.run_validation:
            curr_val_loss = validation(model, local_rank, rank, world_size, test_loader)

        scheduler.step()

        if rank == 0:
            print(f"--> epoch {epoch} completed...entering save and stats zone")

            dur.append(time.time() - t0)
            train_acc_tracking.append(train_accuracy.item())

            if cfg.run_validation:
                val_acc_tracking.append(curr_val_loss.item())

            if cfg.track_memory:
                mem_alloc_tracker.append(torch.cuda.memory_allocated())
                mem_reserved_tracker.append(torch.cuda.memory_reserved())

        if cfg.save_model and curr_val_loss < best_val_loss:
            # update curr best val accuracy

            # save
            if rank == 0:
                print(f"--> entering save model state...")
            save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(
                model, StateDictType.FULL_STATE_DICT, save_policy
            ):
                cpu_state = model.state_dict()
            # states = model.state_dict()
            print(f"saving process: rank {rank}  done w state_dict")
            # dist.barrier()
            # print(f"rank {rank}  done w 2nd barrier")

            if rank == 0:
                print(f"--> saving model ...")
                currEpoch = (
                    "-" + str(epoch) + "-" + str(round(curr_val_loss.item(), 4)) + ".pt"
                )
                model_save_name = "-" + time_of_run + "-" + save_name + currEpoch

                torch.save(cpu_state, model_save_name)

                print(f"--> saved {model_save_name} to disk")
        # announce new val loss record:
        if rank == 0 and curr_val_loss < best_val_loss:

            best_val_loss = curr_val_loss
            print(f"-->>>> New Val Loss Record: {best_val_loss}")

    # init_end_event.record()
    if rank == 0:
        # inner_pbar.close()
        total_training_time = time.time() - training_start_time
        print(f"Total training time = {total_training_time:.2f}")
        print("Times per epoch:")
        for i, val in enumerate(dur):
            print(f"epoch {i}, time {val:.2f}")
        print()

        # memory
        if cfg.track_memory:
            print(f"total memory reserved: {mem_reserved_tracker}")
            print(f"total memory allocated: {mem_alloc_tracker}")

        print(f"Training accuracy: {train_acc_tracking}")
        if cfg.run_validation:
            print(f"Validation accuracy: {val_acc_tracking}")
            print(f"\n Best Val accuracy: {best_val_loss}")

        # memory summary
        if cfg.memory_report and rank == 0:
            print(
                f"CUDA Memory Summary After Last training:\n {torch.cuda.memory_summary()}"
            )
        # print(
        # f"Cuda event elapsed time: {init_start_event.elapsed_time(init_end_event) / 1000}sec"
        # )
        # print(f"{model}")

        # save block
        # save_model = cfg.save_model

        # debug hang
        # runs on all ranks
        # print(f"rank {rank} calling barrier")
        # dist.barrier()
        # print(f"rank {rank} done w barrier, calling state_dict")

    if cfg.save_model:

        # To test resharding, first comment "fsdp_load_state_dict_reshard()" 
        # and run the script with some world size, it will save full states as 
        # baseline and save local states for verification;
        # Then comment "fsdp_save_load_state_dict_no_reshard()" and run the script 
        # with new world size, it will load full states as baseline and load 
        # local states for verification.

        # fsdp_save_load_state_dict_no_reshard(model, model_name, save_name, wrapping_policy, mp_policy, rank, local_rank)

        fsdp_load_state_dict_reshard(model, model_name, save_name, wrapping_policy, mp_policy, rank, local_rank)

    dist.barrier()
    cleanup()


# ------------------ Main functions above ------------


if __name__ == "__main__":

    args = parse_args()

    # seed
    torch.manual_seed(args.seed)
    gpus_per_node = torch.cuda.device_count()

    # torch run start
    fsdp_main(args)

    # cache workaround
    """ dataset_name = "grammar_train.csv"
    full_path_dataset = Path.cwd()/'datasets_grammar'/dataset_name

    temp_full_dataset = load_dataset(
        "csv",
        data_files={
            "train": [full_path_dataset]
        },  # "eval": "grammar_validation.csv"},
        delimiter=",",
    )
    print(f"temp dset loaded in main = len {len(temp_full_dataset)}")
    

    mp.spawn(
        fsdp_main,
        args=(
            gpus_per_node,
            args,
        ),
        nprocs=gpus_per_node,
        join=True,
    )
    """
