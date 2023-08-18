<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a name="readme-top"></a>
<!--




<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->


<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/gardiens/Time-Series-Library_babygarches">
    <img src="images/logos.png.jpg" alt="Logo" width="360" height="360">
  </a>

<h3 align="center">Time-series-Forecasting babygarches</h3>

  <p align="center">
    Deep learning models to predict human skeleton 
    <br />
    <a href="https://github.com/gardiens/Time-Series-Library_babygarches"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/gardiens/Time-Series-Library_babygarches">View Demo</a>
    ·
    <a href="https://github.com/gardiens/Time-Series-Library_babygarches/issues">Report Bug</a>
    ·
    <a href="https://github.com/gardiens/Time-Series-Library_babygarches/issues">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#Pipeline of code">Pipeline of code</a></li>

  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project



The goal of this project is to predict skeleton using Deep-learning architectures especially  using FEDFormers and AutoFormers. It relies heavily on Time-series Library from [thuml]( https://github.com/thuml/Time-Series-Library/tree/main)
Don't forget to give the project a star! Thanks again!

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started
To get a local copy up and running follow these simple  steps.

### Installation
1. Clone the repo 

   ```sh
   git clone https://github.com/gardiens/Time-Series-Library_babygarches.git
   ```
2. install python requirement
  ```py
   pip install requirements.txt
   ```
   if you are working on the dataset NTU RGB+D you may need to download [ffmpeg](https://ffmpeg.org/about.html)
3. If you want to use NTU_RGB download the dataset [here](https://rose1.ntu.edu.sg/dataset/actionRecognition/)
4. run txt2npy. 
the file .npy should be stored in dataset/NTU_RGB+D/numpyed/ and the raw data should be in dataset/NTU_RGB+D/raw/
5. build the csv for the data. it may take a while
 ```py
   python3 build_csv.py
   ```
6. then run the main.py with your argument :)  Somes scripts are provided in the scripts folder. for example:
 ```console
   sh scripts/utils/template_script.sh
   ```
7. You can deep dive on your results with several tools. Videos of some sample are stored in the folder test_results, a dataframe of the loss of each sample is stored in results and you can see your runs in the folder runs thanks to Tensorboard



<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage
this repo provides several features:
- you can preprocess the NTU_RGB+D  dataset efficiently. The implementation is in the folder data_loader
- you can train FEDFormers and AutoFormers on this dataset thanks to exp_Long_Term_Forecast
- you can plot your results.they are stored in test_results after the test of your model.

I added on every folder a readme to help you to grasp what functions are supposed to do. A FAQ is as well available for  any further technical questions. This comments are unfortunately in French.
If you want to use fast some function of this repo, I added a COMMANDE_UTILE.ipynb which is supposed to summarize the needed functions.

<!-- ROADMAP -->
## Roadmap
### Non technical roadmap
- [ ] Insert Categorical value in the prediction.
- [ ] Insert Wavelet Transform

### more technical roadmap 
- [ ]  rewrite the preprocess step to be easier to add new steps.
- [ ] write on Pytorch the preprocessing steps.
- [ ] Ease the fetch of new results and get faster insights on the results. it means to fetch faster the data and have more visual analysis of the models ( gradient/non zero layers..) 




<p align="right">(<a href="#readme-top">back to top</a>)</p>



## Contact


Project Link: [https://github.com/gardiens/Time-Series-Library_babygarches](https://github.com/gardiens/Time-Series-Library_babygarches) 
you can contact me by email ( pierrick.bournez@student-cs.fr ).
Please star if you find this repo useful :) 
<p align="right">(<a href="#readme-top">back to top</a>)</p>




## Pipeline of the code
the code is organised as follow:
1. When you run main.py, it builds an instance of exp/Long_term_forecasting which is the pipeline of the training/test 
2. it find the dataset on [dataset/your_dataset](https://github.com/gardiens/Time-Series-Library_babygarches/tree/master/data_provider) and builds the model in [models/your_model](https://github.com/gardiens/Time-Series-Library_babygarches/tree/master/models). it eventually runs  the training/test code
3. you can fetch the result and have logs on several folder. 
    - In test_results you can see videos of your model after the training session, 
    - in results you have a results_df.csv which is a dataframe that give the loss of every sample of the model. 
    - in runs you have the tensorboards logs of the run.

the setting name is supposed to be a unique ID of each models run. 


<p align="right">(<a href="#readme-top">back to top</a>)</p>

