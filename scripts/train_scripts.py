import os
import random
import shutil
import time
import os 

def gen_commands_lamb(dataset):
    commands = []
    os.makedirs('logs', exist_ok=True)
    cnt_prompts = [1,2,5,10]
    lambs = [0.0,0.1,1.0,0.5,5.0]
    for lamb in lambs:
        for cnt_prompt in cnt_prompts:
            command = f"python -u experiments/clip/TP_training.py --cnt-prompt {cnt_prompt} --lamb {lamb} --dataset {dataset} | tee logs/{lamb}_{cnt_prompt}.log"
            commands.append(command)
    return commands

def run_commands(gpus, commands, call=False, dir="commands", shuffle=True, delay=0.5):
    if os.path.exists(dir):
        shutil.rmtree(dir)
    if shuffle:
        random.shuffle(commands)
        random.shuffle(gpus)
    os.makedirs(dir, exist_ok=True)
    # with open('exp.sb','r') as file:
    #     prefix = file.read()
    with open('scripts/exp.sb','r') as file:
        prefix = file.read()
    fout = open('stop_{}.sh'.format(dir), 'w')
    print("kill $(ps aux|grep 'bash " + dir + "'|awk '{print $2}')", file=fout)
    fout.close()

    n_gpu = len(gpus)
    for i, gpu in enumerate(gpus):
        i_commands = commands[i::n_gpu]
        # prefix = "CUDA_VISIBLE_DEVICES={}".format(gpu)

        sh_path = os.path.join(dir, "run{}.sb".format(i))
        fout = open(sh_path, 'w')
        for com in i_commands:
            print(prefix + com, file=fout)
        fout.close()
        if call:
            os.system("sbatch {}&".format(sh_path))
            time.sleep(delay)

if __name__ == "__main__":
    # commands = gen_commands_unlearn(rerun=False, dense=True)
    # print(len(commands))
    # run_commands(list(range(8)) * 4, commands, call=False,
    #              dir="commands_RL", shuffle=False, delay=0.5)
    # commands = gen_commands_debug_fisher()
    commands = gen_commands_lamb("cifar10")
    print(len(commands))
    run_commands(list(range(20)), commands, call=True,
                 dir="commands", shuffle=False, delay=0.5)