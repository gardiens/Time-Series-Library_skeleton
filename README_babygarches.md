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
    <img src="images/logos.png.jpg" alt="Logo" width="160" height="160">
  </a>

<h3 align="center">Time-series-Forecasting babygarches</h3>

  <p align="center">
    use of Deep Learning models to predict skeletons
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
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

[![Product Name Screen Shot][product-screenshot]](https://example.com)

The goal of this project is to predict skeleton using Deep-learning architectures and especially FEDFormers and AutoFormers. It relies heavily on Time-series Library from thuml( https://github.com/thuml/Time-Series-Library/tree/main)
<p align="right">(<a href="#readme-top">back to top</a>)</p>



<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started
If you want to set up 
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
3. If you want to use NTU_RGB download the dataset [here](https://rose1.ntu.edu.sg/dataset/actionRecognition/)
4. run txt2npy. 
the file .npy should be stored in dataset/NTU_RGB+D/numpyed/ and the raw data should be in dataset/NTU_RGB+D/raw/
5. build the csv for the data. it may take a while
 ```py
   python3 build_csv.py
   ```
6. then run the main.py with your argument :) 




<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

This repository


<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ROADMAP -->
## Roadmap

- [ ] Feature 1
- [ ] Feature 2
- [ ] Feature 3
    - [ ] Nested Feature

See the [open issues](https://github.com/gardiens/Time-Series-Library_babygarches/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!


<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact


Project Link: [https://github.com/gardiens/Time-Series-Library_babygarches](https://github.com/gardiens/Time-Series-Library_babygarches)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/github_username/Time-Series-Library_babygarches
.svg?style=for-the-badge
[contributors-url]: https://github.com/gardiens/Time-Series-Library_babygarches
[forks-shield]: https://img.shields.io/github/forks/github_username/Time-Series-Library_babygarches
.svg?style=for-the-badge
[forks-url]: https://github.com/gardiens/Time-Series-Library_babygarches/network
[stars-shield]: https://img.shields.io/github/stars/github_username/Time-Series-Library_babygarches
.svg?style=for-the-badge
[stars-url]: https://github.com/gardiens/Time-Series-Library_babygarches/stargazers
[issues-shield]: https://img.shields.io/github/issues/github_username/Time-Series-Library_babygarches
.svg?style=for-the-badge
[issues-url]: https://github.com/gardiens/Time-Series-Library_babygarches/issues
[license-shield]: https://img.shields.io/github/license/github_username/Time-Series-Library_babygarches
.svg?style=for-the-badge
[license-url]: https://github.com/gardiens/Time-Series-Library_babygarches/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/linkedin_username
[product-screenshot]: images/screenshot.png
[Next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white
[Next-url]: https://nextjs.org/
[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[Vue-url]: https://vuejs.org/
[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white
[Angular-url]: https://angular.io/
[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00
[Svelte-url]: https://svelte.dev/
[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white
[Laravel-url]: https://laravel.com
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com 
