# CVGAE (A Contrastive Variational Graph Auto-Encoder for Node Clustering)

## Abstract

Variational Graph Auto-Encoders (VGAEs) have been widely used to solve the node clustering task. However, the state-of-the-art methods have numerous challenges. First, existing VGAEs do not account for the discrepancy between the inference and generative models after incorporating the clustering inductive bias. Second, current models are prone to degenerate solutions that make the latent codes match the prior independently of the input signal (i.e., Posterior Collapse). Third, existing VGAEs overlook the effect of the noisy clustering assignments (i.e., Feature Randomness) and the impact of the strong trade-off between clustering and reconstruction (i.e., Feature Drift). To address these problems, we formulate a variational lower bound in a contrastive setting. Our lower bound is a tighter approximation of the log-likelihood function than the corresponding Evidence Lower BOund (ELBO). Thanks to a newly identified term, our lower bound can escape Posterior Collapse and has more flexibility to account for the difference between the inference and generative models. Additionally, our solution has two mechanisms to control the trade-off between Feature Randomness and Feature Drift. Extensive experiments show that the proposed method achieves state-of-the-art clustering results on several datasets. We provide strong evidence that this improvement is attributed to four aspects: integrating contrastive learning and alleviating Feature Randomness, Feature Drift, and Posterior Collapse. 

## Conceptual design

<p align="center">
<img align="center" src="https://github.com/nairouz/CVGAE_PR/blob/master/image_2.png">
</p>

## Some results

### Quantitative 
<p align="center">
<img align="center" src="https://github.com/nairouz/CVGAE_PR/blob/master/image_3.png" >
</p>
<p align="center">
<img align="center" src="https://github.com/nairouz/CVGAE_PR/blob/master/image_4.png" >
</p>

### Qualitative 
<p align="center">
<img align="center" src="https://github.com/nairouz/CVGAE_PR/blob/master/image_1.png">
</p>

## Usage

We provide the code of CVGAE. For each dataset, we provide the pretraining weights. The data is also provided with the code. Users can perform their own pretraining if they wish. For instance, to run the code of CVGAE on Cora, you should clone this repo and use the following command: 
```
python3 ./CVGAE/main_cora.py
```

## Built With

All the required libraries are provided in the ```requirement.txt``` file. The code is buildt with:

* Python 3.6
* Pytorch 1.7.0
* Scikit-learn
* Scipy

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Authors

* **Nairouz Mrabah** - *Grad Student (Université du Québec à Montréal)* 
* **Mohamed Bouguessa** - *Professor (Université du Québec à Montréal)*
* **Riadh Ksantini** - *Professor (University of Bahrain)*

 
## Citation
  
  ```
@article{mrabah2023contrastive,
  title={A contrastive variational graph auto-encoder for node clustering},
  author={Mrabah, Nairouz and Bouguessa, Mohamed and Ksantini, Riadh},
  journal={Pattern Recognition},
  year={2023}
}
  ```
