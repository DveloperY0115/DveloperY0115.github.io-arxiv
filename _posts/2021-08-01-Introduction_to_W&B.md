---
layout: post
title: "Brief Introduction to Logging with Weights & Biases"
background: '/assets/post-images/Introduction_to_W&B/background.png'
---

# Introduction

This post is written to serve as the minimal starting point, including basic concepts and ready-to-run code snippets, for you when integrating Weights & Biases (I will call it W&B after all) into your machine learning workflow. 

As you can notice from the word "minimal", this post is just a very condensed summary covering only a small portion of the entire W&B documentation. Therefore, I highly encourage you to check out [the full documentation](https://docs.wandb.ai/) to learn more after finishing this.

In a nutshell, W&B is **a tool for experiment tracking, dataset versioning, and model management** which we, ML researchers / engineers, do twenty-four-seven. If you are already familiar with TensorBoard, I bet you can easily get most of the contents and start writing codes immediately. Even if you aren't, W&B is quite straightforward to use yet incredibly powerful, so I strongly recommend you to take some time and go over this material (it won't take that long!).

For explanation, and to give you a sense of adopting W&B in real-world ML projects, I will use my implementation of *PointNet (PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation, Charles R. Qi et al., CVPR 2017)* as an exempler. Note that all code snippets in this post are from my [*torch-pointnet* repository](https://github.com/DveloperY0115/torch-pointnet).

# Initializing W&B

## Sign Up & Sign In

Don't have an account for Weights & Biases? You can easily make one, or directly link your Github account to it.

After signing up, you need to install *wandb* module to your Python environment. Everything we wil do after all assumes that you've installed this module successfully. And in fact, this module is all you need. Installation is surprisingly simple - just run the following command in your shell.

~~~bash
pip install wandb
wandb login
~~~

## Starting a New Run

If you have experience with TensorBoard, you would remember that we need to initialize a *SummaryWriter* object which is then used for logging throughout the rest of your program. W&B has no difference, but is more elegant in some sense compared to TensorBoard. Running a single line of code will do everything for us.

```python
# imports
# import torch, numpy, etc
import wandb
# ...

# def main():
# bla bla bla

# and many other functions
# bla bla bla

if __name__ == "__main__":
	
	# initialize W&B
	wandb.init(project="torch-pointnet")

	# run main function
	main()
```

The above code snippet shows the overall structure of my *train.py* file, which defines various control flows involving the training of PointNet. As you can see, *wandb* is initialized before running the main function which contains a number of *wandb.log* (will be explained soon) calls for tracking important quantities (loss, accuracy, etc) during the experiment. 

# Logging & Model Tracking

## Log (Almost) Everything You Want

*wandb.log* is a function that we use for logging. You can log almost everything you can imagine in various ML workflows - numbers, images, audio, video, HTML, Histogram, 3D data, and much more. You can check the supported types [here](https://docs.wandb.ai/guides/track/log).

In my case, I use this for tracking:

- Training loss
- Test loss
- Test accuracy
- Visualization of test result (for qualitative analysis)

For instance, in the function *train_one_epoch* that carries out optimization for a single epoch, I simply write:

```python
def train_one_epoch(network, optimizer, scheduler, device, train_loader, epoch):
	
	# forward propagation
	# bla bla bla

	# back propagation
	# bla bla bla

	# update
	# bla bla bla

	# log data
	wandb.log({"Train/Loss": avg_loss}, step=epoch)
```

The quantity of interest is packed into a Python dictionary as a *(key, value)* pair where **the key becomes the title of a plot**, and **the value represents the actual metric** to be recorded. Notice how simply the logging works. 

Of course, I can put as many kinds of data into a dictionary and log them at once.

```python
def run_test(network, device, test_loader, epoch):
	
	# forward propagation

	# compute metrics used for testing

	# log data
	wandb.log(
        {"Test/Loss": loss, "Test/Accuracy": accuracy, "Test/Result": wandb.Image(fig)}, step=epoch
    )
```

(*Disclaimer: Logging image data isn't working as intended at the time I write this post. I'll try to resolve this issue ASAP.)*

As a result, you can see this beautiful plot of training loss decreasing nicely over time. What makes this even cooler is that **you can check the results in real time at the control panel through your web browser** (both PC and mobile) without logging into your remote machine.

<center>
    <img class="img-fluid" src="/assets/post-images/Introduction_to_W&B/fig1.png">
</center>

## Anatomy of Your Model with W&B

Having problem with exploding (or vanishing) gradients and don't know which part~~(s)~~ of your model is ~~(are)~~ in trouble? Using W&B, you can easily diagnose the problem. To make W&B pay attention to the internal of your model, simply wrap the model after initializing it.

~~~python
		
def main():
		# check GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		
		# model & optimizer, schedulers
    network = PointNetCls().to(device)

    # make W&B track model's gradient and topology
    wandb.watch(network, log_freq=100, log_graph=True)

		# and the rest of main..
~~~

W&B gives you the freedom of choosing which aspects of your network are going to be tracked. In the snippet above, I decided to record the gradients at all layers of my model every 100 steps and track the computational graph formed during training.

Just like logged metrics, this information will show up in your project dashboard.

<center>
    <img class="img-fluid" src="/assets/post-images/Introduction_to_W&B/fig2.png">
</center>

# Check System Utilization

You might want to check whether you're pushing your machine to its limit. Thankfully, W&B automatically tracks various statistics of the system throughout a session - GPU memory usage, power consumption, number of CPU threads in use, etc.

<center>
    <img class="img-fluid" src="/assets/post-images/Introduction_to_W&B/fig3.png">
</center>

# Hyperparameter Tuning with Sweep

Personally, I think this is **the coolest feature of W&B**. W&B provides a powerful yet easy-to-use tool for hyperparameters tuning with fancy visualizations that helps users intuitively compare different combinations of them.

In this example, although my model has only few hyperparameters associated with its structure and training, I'll try to find the sweet spot for learning rate and batch size.

In order to find the optimal combination of hyperparameters, we need to tell W&B,

- **which hyperparameters to adjust**
- **what are the possible values (i.e. explorable space) for such hyperparameters**
- **which metric we want to opimize further through hyperparameter tuning**

## Tell W&B a list of adjustable variables

In my case, I personally prefer using *argparse* for tweak variables for training. As you can see at the beginning of my *train.py,* I store the set of arguments in variable *args*.

```python
parser = argparse.ArgumentParser(description="Parsing argument")
parser.add_argument("--beta1", type=float, default=0.9, help="Beta 1 of Adam optimizer")
parser.add_argument("--beta2", type=float, default=0.999, help="Beta 2 of Adam optimizer")
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for optimizer")
parser.add_argument("--step_size", type=int, default=100, help="Step size of StepLR")
parser.add_argument("--gamma", type=float, default=0.99, help="Gamma of StepLR")
parser.add_argument("--num_epoch", type=int, default=100, help="Number of epochs")
parser.add_argument("--num_iter", type=int, default=100, help="Number of iteration in one epoch")
parser.add_argument("--batch_size", type=int, default=64, help="Size of a batch")
parser.add_argument("--num_worker", type=int, default=8, help="Number of workers for data loader")
parser.add_argument("--out_dir", type=str, default="out", help="Name of the output directory")
parser.add_argument(
    "--save_period", type=int, default=100, help="Number of epochs between checkpoints"
)
args = parser.parse_args()
```

Then I need to inform W&B what the controllable variables are. All I need to do this is simply passing the set of parsed arguments at the time of initialization. Note that this is one way of doing this, and you can see other methods [here](https://docs.wandb.ai/guides/sweeps/quickstart).

```python
if __name__ == "__main__":

    wandb.init(project="torch-pointnet", config=args)

    main()
```

And in function *main*, I replace all occurences of *args.** to *config[*]*. Here, the object *config* holds the variables that we informed to W&B at initialization.

```python
def main():
    
    config = wandb.config

    # ...

    optimizer = optim.Adam(network.parameters(), betas=(config["beta1"], config["beta2"]), lr=config["lr"])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config["step_size"], gamma=config["gamma"])

		# ...

    train_loader = DataLoader(train_data, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_worker"])
    test_loader = DataLoader(test_data, batch_size=config["batch_size"], shuffle=True, num_workers=0)

		for epoch in tqdm(range(config["num_epoch"]), leave=False):
        avg_loss = train_one_epoch(network, optimizer, scheduler, device, train_loader, epoch)

				# rest of main..
```

## Configure Sweep

Next, I wrote a YAML file to specify the hyperparameters that I want to sweep over, the way how to explore the hyperparameter space, and the set of possible values for each hyperparameter.

```yaml
program: train.py
method: grid
metric:
  name: Test/Loss
  goal: minimize
parameters:
  lr: 
    values:
    - 0.1
    - 0.01
    - 0.001
    - 0.0001
  batch_size:
    values:
    - 32
    - 64
    - 128
    - 256
```

In particular, I'm telling W&B: 

- that the training routine is defined at the file *train.py*
- to use "grid" method (examine all possible combinations of hyperparameter values) for exploration
- to "minimize" the metric "Test/Loss". **One important note is that you should log a quantity with name "Test/Loss" somewhere in the training routine.** And I did it previously in the function *run_test*.
- variables *"lr"* and *"batch_size"* are the ones that can be modified for each different run. And each of them has a set of possible values (e.g. learning rate (*lr*) can be one of 0.1, 0.01, 0.001, 0.0001)

## Initialize & Run Sweep

After setting up a sweep configuration in **.yaml* file, run the following command to initialize the sweep:

```bash
wandb sweep *.yaml
```

W&B will automatically set up things for you and give you a *sweep ID* which specifies that exact sweep to be run. **Copy that and use it in the next step.**

## Launch Agent(s)

What makes sweep feature more powerful is that multiple machines or processes can contribute to the same sweep. This means, you can concurrently test different combinations of hyperparameters across your devices or processes in a single machine. As long as you share the same sweep ID, W&B will distribute tasks to agents participating in the sweep and all you have to do is just waiting for the result.

The below image shows the result of each trial in the middle of sweeping (i.e. it was still on going). As soon as you see the plot at the bottom, you will immediately notice that the cases using large learning rate tend to give higher test losses while batch size seems to have no effect on test loss.

<center>
    <img class="img-fluid" src="/assets/post-images/Introduction_to_W&B/fig4.png">
</center>

# Summary

In this post I briefly introduced only a few of, yet fundamental functionalities of W&B. These include:

- How to set up W&B for your project
- Logging metrics and various kinds of data using *wandb.log*
- Analyzing model with *wandb.watch*
- Checking statistics of system usages collected by W&B
- Sweep - a powerful way of tuning hyperparameters & visualizing results to gain insights

I hope this post was helpful & comprehensible to you, would be glad if you can benefit from it, work in more productive way. Furthermore, as I mentioned in the introduction, be sure to check out the official documentation for more details and things that were not covered here.