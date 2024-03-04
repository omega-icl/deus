# **DEUS Python Package**


## **1. Goal**

DEsign under Uncertainty using Sampling techniques (DEUS for short) offers Nested Sampling based methods for solving the following problems:
- parameter estimation (PE)
- parameter set membership (PSM)
- design space characterization (DS)



## **2. Building the package**
As a package development collaborator you want to allow others to use the 'deus' package. More specifically, you would like to send to the users an archive file which they can easily use to install the package on their machines. Below you find how to do that.

## **2.1 Building a source distribution**
We create a deus-<version>.tar.gz archive which can be sent to any user and (s)he can install the 'deus' package simply by running in Command Prompt (or equivalent) the following command 'pip install deus-<version>.tar.gz'.

How to do that?
Step 1: Open Command Prompt (or equivalent) and go to 'src' directory ('cd .../src').
Step 2: Run 'python setup.py sdist'. This command means that you want Python to run setup.py file in order to create a source distribution archive. A 'dist' folder will be created within 'src' directory. Also an egg-info folder will be created.
Voila! In 'dist' directory you will find the file 'deus-<version>.tar.gz' that can be used by any user that wants to install the package.

What if you want to delete the 'dist' and egg-info folders automatically?
Step 1: Open Command Prompt (or equivalent) and go to 'src' directory ('cd .../src').
Step 2: Run 'python setup.py clean'. This command means that you want Python to run setup.py file in order to clean the 'dist' and egg-info folders.
Observation: It is recommended to first 'clean' and then to use sdist option when you want to build a package source distribution.



## **3. Installing the package**

## **3.1. Installing from a source distribution**
Step 1: Open Command Prompt (or equivalent) and go to the directory containing the 'deus-<version>.tar.gz' archive.
Step 2: Run 'pip install deus-<version>.tar.gz'. pip will install 'deus' package in the standard place, i.e. <Python-distribution-home>/Lib/site-packages/.
That's it! Now you can import 'deus' in any project you want to use it.

Observation: If you go to <Python-distribution-home>/Lib/site-packages/ then you will find the folder 'deus' which contains the package.


## **4. Uninstalling the package**
Step 1: Open Command Prompt (or equivalent) anywhere.
Step 2: Run 'pip uninstall deus'. pip will uninstall 'deus' package, so the <Python-distribution-home>/Lib/site-packages/ will no longer contain the 'deus' folder.
