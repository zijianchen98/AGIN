<div align="center">

<div>
<a href="https://github.com/zijianchen98/AGIN"><img src="https://visitor-badge.laobi.icu/badge?page_id=zijianchen98/AGIN"/></a>
    <a href="https://github.com/zijianchen98/AGIN"><img src="https://img.shields.io/github/stars/zijianchen98/AGIN"/></a>
    <a href="http://arxiv.org/abs/2312.05476"><img src="https://img.shields.io/badge/Arxiv-2312:05476-red"/></a>
    <a href="https://github.com/zijianchen98/AGIN"><img src="https://img.shields.io/badge/Database-Release-green"></a>
    <a href="https://github.com/zijianchen98/AGIN"><img src="https://img.shields.io/badge/Awesome-AGIN-orange"/></a>
</div>

<h1>Exploring the Naturalness of AI-Generated Images</h1>

_Do AI-generated Images look natural?_

<div>
    <a href="https://scholar.google.com.hk/citations?hl=zh-CN&user=NSR4UkMAAAAJ" target="_blank">Zijian Chen</a><sup>1</sup>,
    <a href="https://scholar.google.com.hk/citations?hl=zh-CN&user=nDlEBJ8AAAAJ" target="_blank">Wei Sun</a><sup>1*</sup>,
    <a href="https://scholar.google.com.hk/citations?hl=zh-CN&user=wth-VbMAAAAJ" target="_blank">Haoning Wu</a><sup>2</sup>,
    <a href="https://scholar.google.com.hk/citations?hl=zh-CN&user=QICTEckAAAAJ" target="_blank">Zicheng Zhang</a><sup>1</sup>,
    <a href="https://scholar.google.com.hk/citations?hl=zh-CN&user=QICTEckAAAAJ" target="_blank">Jun Jia</a><sup>1</sup>,
    <a href="https://minxiongkuo.github.io/" target="_blank">Xiongkuo Min</a><sup>1</sup>,
    <a href="https://ee.sjtu.edu.cn/FacultyDetail.aspx?id=24&infoid=66&flag=66" target="_blank">Guangtao Zhai</a><sup>1*</sup>,
    <a href="https://ee.sjtu.edu.cn/FacultyDetail.aspx?id=14&infoid=66&flag=66" target="_blank">Wenjun Zhang</a><sup>1</sup>
</div>

<div>
  <sup>1</sup>Shanghai Jiao Tong University, <sup>2</sup>Nanyang Technological University
</div>   

<div>
<sup>*</sup>Corresponding author. 
   </div>

<a href="https://arxiv.org/abs/2312.05476"><strong>Paper</strong></a>

<div style="width: 80%; text-align:center; margin:auto;">
<img style="width:100%" src="intro.jpg"></div>

</div>

> Abstract: We take the first step to benchmark and assess the visual naturalness of AI-generated images. First, we construct the **AI-Generated Image Naturalness (AGIN) database** by conducting a large-scale subjective study to collect human opinions on the **overall naturalness** as well as perceptions from **technical and rationality perspectives**. AGIN verifies that naturalness is universally and disparately affected by both technical and rationality distortions. Second, we propose the **Joint Objective Image Naturalness evaluaTor (JOINT)**, to automatically learn the naturalness of AGIs that aligns human ratings. Specifically, JOINT imitates human reasoning in naturalness evaluation by jointly learning both technical and rationality perspectives. Experimental results show our proposed JOINT significantly surpasses baselines for providing more subjectively consistent results on naturalness assessment.


# AI-Generated Image Naturalness (AGIN) Database

## Introduction
The proposed AGIN includes two perspectives for naturalness assessment: technical (T) and rationality (R), each of which contains 5 dimensions:

- For technical perspective (T), we consider specific image attributes (_**Luminance**, **Contrast**_) that have high correlations of naturalness. _**Detail**, **Blur**_ and a common issue introduced by generative models _(**Artifacts**)_ are also considered.
- For rationality perspective (R), we contribute 5 new factors: _**Existence**_, _**Color**_, _**Layout**_, _**Context**_, and _**Sensory Clarity**_ to evaluate the rationality of AI-generated images.
- Please refer to our paper for more detailed explaination.


## Database Downloads
OneDrive and Baidu Netdisk are available now!

* OneDrive: [downloading link](https://1drv.ms/u/s!AgLwywHLkSEMk1Jmi_PLz-JPnaxq?e=qLx41h)
* Baidu Netdisk: [downloading link](https://pan.baidu.com/s/1DWZEMH6fPndaD9UhJbylMQ) (password: riha)


## Collecting AI-generated Images
The adopted image generation methods and related works are listed in another repository [Awesome-AI-Generated-Image-Tasks](https://github.com/zijianchen98/Awesome-AI-Generated-Image-Tasks)

## Citation
Please feel free to cite our paper if you use the AGIN database in your research:
```
@article{chen2023exploring,
  title={Exploring the Naturalness of AI-Generated Images},
  author={Chen, Zijian and Sun, Wei and Wu, Haoning and Zhang, Zicheng and Jia, Jun and Min, Xiongkuo and Zhai, Guangtao and Zhang, Wenjun},
  journal={arXiv preprint arXiv:2312.05476},
  year={2023}
}
```