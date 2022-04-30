<br />
<div align="center">

  <h3 align="center">
MITNET</h3>

  <p align="center">
   A Two-Stage Deep Learning Approach for Mitosis Recognition
    <br />
    <br />
    <a href="http://212.156.134.202:4481/">View Demo</a>

  </p>
</div>



### Abstract

Mitosis assessment of breast cancer has a strong prognostic importance and is visually evaluated by pathologists.
The inter, and intra-observer variability of this assessment is high. In this paper, a two- stage deep learning approach,
named MITNET, has been applied to automatically detect nucleus and classify mitoses in whole slide images (WSI) of breast cancer. 
Moreover, this paper introduces two new datasets. The first dataset is used to detect the nucleus in the WSIs, which contains 139, 124 annotated nuclei 
in 1749 patches extracted from 115 WSIs of breast cancer tissue, and the second dataset consists of 4908 mitotic cells and 4908 non-mitotic cells 
image samples extracted from 214 WSIs which is used for mitosis classification. The created datasets are used to train the MITNET network, which consists of 
two deep learning architectures, called MITNET-det and MITNET-rec, respectively, to isolate nuclei cells and identify the mitoses in WSIs. 
In MITNET-det architecture, to extract features from nucleus images and fuse them, CSPDarknet and Path Aggregation Network (PANet) are used respectively,
and then a detection strategy using You Look Only Once (scaled-YOLOv4) is employed to detect nucleus at three different scales. In the classification part,
the detected isolated nucleus images are passed through proposed MITNET-rec deep learning architecture, to identify the mitosis in the WSIs. Various deep learning 
classifiers and the proposed classifier are trained with a publicly available mitosis datasets (MIDOG and ATYPIA) and then validated over our created dataset. 
The results verify that deep learning based classifiers trained on MIDOG and ATYPIA have difficulties to recognize mitosis on our dataset which shows 
that the created mitosis dataset has unique features and characteristics. Besides this, the proposed classifier outperforms the state-of-the-art classifiers
significantly, and achieves a 68.7% F1-score and 49.0% F1-score on the MIDOG and the created mitosis datasets, respectively. Moreover, 
the experimental results reveal that the over-all proposed MITNET framework detects the nucleus in WSIs with high detection rates and recognizes the mitotic
cells in WSI with high F1-score which leads to the improvement of the accuracy of pathologists’ decision.


<p align="center" width="100%">
    <img src="/flow.png">
</p>



## Citation
S. Cayir, et al., "MITNET: A Novel Dataset and A Two-Stage Deep Learning Approach for Mitosis Recognition in Whole Slide Images of Breast Cancer Tissue", under revision, Neural Computing and Applications.


<!-- CONTACT -->

## Contact

Sercan Çayır - sercan.cayir@virasoft.com.tr

<!-- CONTRIBUTING -->

## Contributing

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also
simply open an issue with the tag "enhancement". Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

