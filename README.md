# generative join

## Setup Instructions

Approximate setup time: 20 minutes.  (5 minutes of work + 15 minutes of installation in the background.)

### Step 1: Install the `gcloud` CLI

First, you'll need to [install the Google Cloud command line
tools](https://cloud.google.com/sdk/docs/install).

- Follow the instructions on the [installation
page](https://cloud.google.com/sdk/docs/install)
- run `gcloud init` as described [in this
  guide](https://cloud.google.com/sdk/docs/initializing) and configure the tool
  with the ID of your new Cloud project.

### Step 2: Connect to the configured gcloud instance

We've created a Google Cloud instance with an attached GPU for you to use with
this development workflow. Run the following snippet to set a few environment
variables:

```bash
export ZONE="us-central1-a"
export PROJECT="probcomp-caliban"
export INSTANCE_NAME="generative-join"
```

Configure the `ssh` credentials required to reach the instance with the `gcloud`
cli:

```bash
gcloud compute config-ssh --project $PROJECT
```

You should now be able to log in to the
instance with the following command:

```bash
ssh $INSTANCE_NAME.$ZONE.$PROJECT
```

If you have any trouble, visit the [cloud
console](https://console.cloud.google.com/compute/instances?project=probcomp-caliban)
and make sure that the `generative-join` instance is running.
If it's not, select
the instance's checkbox and click "Start" to boot the machine up again for use.
After a couple minutes, the instance should be running and the ssh command should work.

### Step 3: Set up git access and clone the repository

Once you are connected to the VM, add a github ssh key to the VM,
so that you can clone the repository (and push changes back to it)
from the machine.

1. run `ssh-keygen -t ed25519 -C "your_email@here.com" (filling in your email address).
    This will generate an SSH key.  I recommend you use a passphrase if you are using
    a cloud machine that probcomp gave you access to (since the probcomp members
    who set up the machine for you have access to it too).
2. Run `eval $(ssh-agent -s)` and then `ssh-add ~/.ssh/id_ed25519` to add the key to the ssh agent.
3. Run `cat ~/.ssh/id_ed25519.pub` and copy the output to your clipboard.
4. Visit the [github ssh key settings](https://github.com/settings/keys) and add the key to your account.
    (Click "New SSH Key" in the top right corner, and copy in the public key you just generated.)

Now you can clone the repository:
```
git clone git@github.com:probcomp/generative_join.git
```

### Step 4: Connect to VS Code

To run the code, we have set up a development VM which is pre-configured
to load in the dependencies needed to run Bayes3D.
This VM will run on the google cloud server (and we can later configure it to run on local hardware).
We have set this up so that you can connect to this VM
directly from VS Code on your desktop,
to develop and run code through it.

Start VS code on your desktop.  If you don't have it already, install
the `Dev Containers` VS Code extension, by Microsoft.
(You can do this by clicking the Extensions icon on the left, which looks like
three connected squares with a disconnected square in the top right.  In the Extensions tab, search for "Dev Containers".)

Once you have Dev Containers installed, click on the `><` icon on the bottom left
corner of VS Code to get the `Connect to Host` menu.
Choose your VM, which should be among
the choices due to using `config-ssh`, and connect. Once on your cloud
VM, open the `generative_join` folder you created by cloning the git repository.

### Step 5: Run the code through VS Code

At this point, you should be able to run the code in the `generative_join`
through your VS Code window.

To run the tutorial notebook, open it
and run the cells!


You are welcome push changes to the code to branches on github.

### Step 6: Shut down the VM

When you're done working, visit the [cloud
console](https://console.cloud.google.com/compute/instances?project=probcomp-caliban),
select the `generative-join` image and hit the "Stop" button that appears in the
pop-up menu.

When you're ready to work again, visit the [cloud
console](https://console.cloud.google.com/compute/instances?project=probcomp-caliban)
and hit "Start" to boot the machine up again for use.

