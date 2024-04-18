---
layout: distill
title: Image Inpainting using Deep Learning 
description: Image Inpainting using Deep Learning 
tags: distill formatting
featured: true
date: 2022-02-20

toc:
  - name: Problem Statement
  - name: Related works
  - name: Traditional Methods
  - name: Learning-based Techniques
  - name: Vision Transformers
  - name: Thesis Project
  - name: Gated Convolution  
  - name: TransGAN
#   subsections:
#     - name: Gated Convolution   
    # subsections:
    #   - name: Dataset and Preprocessing
    #   - name: Model
    #   - name: Architecture
    #   - name: Training Information
    #   - name: Results
    #   - name: Limitations
    # - name: TransGAN
    # subsections:
    #   - name: Dataset and Preprocessing
    #   - name: Model
    #   - name: Architecture
    #   - name: Training Information
    #   - name: Results
    #   - name: Limitations
  - name: Transformer vs CNN
  - name: Results
---

## Problem statement

Image Inpainting is a technique for reconstructing an image's missing pixels or missing regions. This approach is used to either delete an object from a picture or recover the image's corrupted elements. These are being used in the form of 3D image inpainting in medical images and AR/VR technologies in recent trends. It's a significant solution to remove image occlusions. Object removal, picture restoration, modification, re-targeting, compositing, and image-based rendering are among the imaging and graphics applications that use it. Our main goal is to construct an efficient and robust inpainter by leveraging the most recent and proven architectures and techniques in computer vision. 

## Related works

Without deep learning, there is a whole realm of computer vision to explore. Object detection was still possible before Single Shot Detectors (SSDs) were invented (although the precision was not anywhere near what SSDs are capable of). In the same way, there are a few traditional computer vision approaches for image inpainting. Inpainting techniques can be divided into two categories: traditional and learning-based. We looked at methods from both classes in this part, as well as provided further detail on the techniques used in the comparison.

### Traditional Methods

The traditional image inpainting approaches, also known as traditional or non-learning-based approaches, perform the inpainting process using specific algorithms and can be divided into two categories: diffusion-based methods and patch-based methods. Traditional methods do not necessitate any training. As a result, it's a good technique for solving easy problems. When the masked area in the image begins to develop a proportionally huge area, traditional methods do not perform well. Given that these algorithms infer from pixels near the missing region, this result is expected.

**Patch-Based Methods** 

In patch-based image inpainting, the undamaged regions of the image are used to build the region to be filled. The goal of this method is to achieve the maximum possible patch similarity level. While patch-based approaches are effective, they assume that the information in the missing portion of the image is stored somewhere else in the image. Patch-based approaches require more processing power because they are methods that involve searching and comparing all of the time.

1. Patch-based Texture Synthesis
2. Image Melding

**Diffusion based method**

The first type of digital image inpainting technique was diffusion-based inpainting. The diffusion process covers the image. This inpainting method was developed by integrating principles from classical fluid dynamics with partial equations, and it is still one of the most often used traditional methods today. The assumption is that the image's edges are continuous.

1. Navier-Stokes Method 
1. Fast Marching Method

### Learning-based Techniques

Deep learning algorithms have been used for image inpainting, as they have been for many other computer vision challenges. Deep learning approaches are becoming more popular as time goes on because, when compared to older methods, they produce better results in complex problems. The fundamental reason for this is the generation of large-scale datasets that will allow deep methods to be trained, as well as the processing capacity that will allow these deep methods to be trained.

**CNN-based Methods** 

Convolutional neural network structures, which are well-known for their grid-like layer topology and high success in computer vision studies, are also used in image inpainting research and produce excellent results. Many architectures are designed specifically for inpainting work. The architecture of the U-Net One of these is the U-Net architecture, which was used in the Shift-Net inpainting study. This architecture uses convolutional layers to combine an image and a mask that shows where the missing portions are. In a symmetrical design, it concatenates each layer's output with the output of the corresponding layer of the same size 24. In terms of generated picture structure and fine detail, the outcomes of this structure are really successful.


**GAN-based Methods**

Various deep learning methods have been tested over time, and it has been determined that some methods produce better results. CNN and GAN-based algorithms, it may be stated, are superior at analyzing realistic results in images. Methods are based on the GAN concept.

An encoder-decoder design first creates a feature space using its encoder, according to the research Context Encoder: Feature Learning by Inpainting. The decoder then uses that area to generate a realistic inpainted output image. Furthermore, by integrating two losses, vastly better and more realistic results can be obtained.

In addition to this research, many techniques for resolving various difficulties have been offered. Using fewer downsampling layers is recommended to avoid blurry regions in the produced images. Furthermore, instead of fully connected layers, dilated convolution layers were used. Unfortunately, because of the relatively sparse filters created by the dilation element, these adjustments resulted in longer training times. SC-FEGAN used gated and dilated gated convolutional layers in an image completion study.

**Edge Connect**

The model's working logic is divided into four phases. In a nutshell, each iteration's input image is directed to the dataset script. The method of detecting an edge and extracting the image from the colour channels is conducted in this dataset script. The edge and grey image are given to the edge model as input from the parts obtained in this stage. The output is reverse masked before even being added to the original portions. As a result, we only get the part of the output picture that is produced around the masked region. The RGB image is then given to the inpainted model and reversely masked to use the edge information obtained from this section.

### Vision Transformers

When compared to convolutional neural networks (CNN), Vision Transformer (ViT) achieves remarkable results while using less computational resources for pre-training. When training on smaller datasets, Vision Transformer (ViT) does have a weaker inductive bias than convolutional neural networks (CNN), resulting in a greater reliance on model regularisation or data augmentation (AugReg).

The ViT is a visual model based on a transformer's architecture, which was originally designed for text-based activities. The ViT model represents an input image as a series of image patches, similar to how word embeddings are represented in text when utilizing transformers, and predicts class labels for the image directly. When trained on enough data, ViT outperforms an equivalent state-of-the-art CNN using 4x fewer Computational resources.

The optimizer's decision, network depth, and dataset-specific hyperparameters all affect the performance of a vision transformer model. CNN’s are less difficult to improve than ViT. The difference between a pure transformer and a CNN front end is to marry a transformer to a CNN front end. A 16\*16 convolution with a 16 stride is used in the conventional ViT stem. A 3\*3 convolution with stride 2 on the other hand, improves stability and precision.

In machine learning, a transformer is a deep learning model that uses attention mechanisms to weigh the significance of each part of the input data differently. Machine learning transformers are made up of many self-attention layers. Natural language processing (NLP) and computer vision are two AI subfields that employ them extensively (CV).

## Thesis Project

We have chosen a paper to work on [6]. We trained our model using the techniques Gated convolution and SN patch-based GAN. Our team with our mentor has agreed to work on the paper “Free-form image inpainting using gated convolution”.

## Gated Convolution

#### Dataset and Preprocessing

**Used Dataset: Places2**

Link for Places2 dataset: <http://places2.csail.mit.edu/>

We evaluated the image inpainting system by using the Places2 dataset in which it has many images and these images are of wide ranges of varieties. Here we have taken a set of 15k images in the ratio of 70:30 for the training of the model and for testing of the model. Here the selected images are of different features which are vast in numbers. In the future, we will train faces for the CelebA-HQ dataset.

<div class="row justify-content-center">
    <div class="col-sm-8 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/image-inpainting/1.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Images present in Places2 Dataset 
</div>

The above figure1 depicts some of the images present in the Places2 dataset. 

This dataset has undergone transformations mainly random cropping, horizontal flip, and vertical flip.

Flists are the list of paths of the images from the datasets which are shuffled. These Flists are used to open the images during training and convert them into different resolutions for easy training.

During the training process, we have generated 100 free-form masks using an algorithm mentioned in the paper and used them for the training.

#### Model

The model has the Generator and Discriminator parts with the Self-attention included. The following parts of the report will explain in detail the different parts. In the Generator, we took encoder-decoder networks for obtaining the coarse image and refined image. The steps with the refined image contain Self-attention. Next, we do pass through it through the discriminator where the spectral normalization is used for attaining stability. Here the gated convolution is used instead of convolution to treat valid and invalid pixels separately. 

<div class="row justify-content-center">
    <div class="col-sm-8 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/image-inpainting/2.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Overview of framework with gated convolution
</div>

#### Architecture

Generative Adversarial Network(GAN) is used here to fill out the masked regions of the image. This encoder-decoder network has two parts namely Generator and Discriminator. The generator’s work is to generate the images close to the target. Discriminators’ work is to distinguish the difference between real and generated images. This Generator and Discriminator generally follow the minimax theorem in terms of losses. The GAN  we used in the model is called the Spectral Normalisation Patch GAN where we used spectral normalization for ensuring stability in the Discriminator part. This is a conditional GAN where we pass both the input and the target values to attain the results. The loss functions for the Generator and the Discriminator are as follows:

<b>Generator          :  L<sub>G</sub> = −E <sub>z~Pz(z)</sub> [D<sup>sn</sup>(G(z))]</b> 

<b>Discriminator   :  L<sub>D</sub> = E<sub>x~Pdata(x)</sub> [ReLU( 1− D<sup>sn</sup>(x))] +E<sub>z~Pz(z)</sub> [ReLU(1 + D<sup>sn</sup>(G(z)))]</b> 

#### Training Information

Used 50 epochs for training our model. Used 2 lakh images as training samples and 25K as testing samples. Captured the loss functions and SSIM values.

**Difficulties:** It took a lot of time to train on our dataset. Google Colab frequently 

got disconnected.

**GPU used:** Google Colab's Free GPU


#### Testing using different images

**Rectangular mask vs Free-form mask** 

Taking a mask as an input we have used different kinds of masks from a set to produce outputs. Mainly here when trained we get outputs when we choose the mask as a rectangular mask and mask as a free-form mask.  

**Ground Truth                   Mask                   Completed Image**

<div class="row justify-content-center">
    <div class="col-sm-8 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/image-inpainting/3.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Input given with free-form mask 
</div>

<div class="row justify-content-center">
    <div class="col-sm-8 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/image-inpainting/4.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Results of input given with free-form mask
</div>


**Ground Truth                         Mask                      Completed Image**

<div class="row justify-content-center">
    <div class="col-sm-8 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/image-inpainting/5.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Input given with rectangular-mask	
</div>

As shown the outputs are as follows where figure 6,figure7 shows the ground truth and completed image produced by giving a free-form mask as an input.

And figure 8 shows the ground truth and completed image produced by using a rectangular mask as an input.

**About different image resolutions produced**

These are the images produced when trained. Here we have got outputs in the resolutions of different sizes such as 64x64, 128x128, and 256x256 respectively.

By looking at the results we can see that we got a clear image resolution in the 256x 256 images. Hence we will consider taking 256x256 resolution images further.

<div class="row justify-content-center">
    <div class="col-sm-9 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/image-inpainting/6.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Image produced in 64x64 size
</div>

<div class="row justify-content-center">
    <div class="col-sm-9 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/image-inpainting/7.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Image produced in 128x128 size	
</div>

<div class="row justify-content-center">
    <div class="col-sm-9 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/image-inpainting/8.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
   Image produced in 256x256 size	
</div>


In the above pictures, Figure 9 represents 64x64 image outputs, Figure 10 represents 128x128 image outputs and Figure 11 shows 256x256 image outputs.


#### Results

We evaluated our model on the Places2 dataset. For testing it runs at 0.22 seconds per image on Google Colab’s GPU for images of resolution 256x256 on average, regardless of hole size. Our model has mainly 3 losses which are Generator loss [3.6], Discriminator loss[3.6], Reconstruction loss taken as L1 for original image and reconstructed image. The reconstruction loss is taken as the average for the coarse image and refined image with and without mask regions.

By recording the SSIM values for the dataset we acquired a value of approximately 0.77   SSIM as shown in the graph figure13 below.

Below are some of the results produced after training the model

<div class="row justify-content-center">
    <div class="col-sm-10 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/image-inpainting/9.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Result produced with the free-form mask
</div>

<div class="row justify-content-center">
    <div class="col-sm-10 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/image-inpainting/10.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Image showing multiple image outputs
</div>

Figure 14 has the images where it has a ground truth and input is given with rectangular masks which produce respective outputs. Figure15 depicts the final image outputs produced when trained with different masks.


|**Model** |**Mask**|**Dataset**|**FID**   |**SSIM**  |
| :- | :- | :- | :- | :- |
|Gated Convolution|Free-form, rectangle|Places 2|91|0\.87|
|TransGAN|Random Rectangle |CelebA-HQ|130|0\.56|



#### Limitations

<div class="row justify-content-center">
    <div class="col-sm-8 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/image-inpainting/11.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Results produced after training with different masks
</div>

Our model didn’t produce great results for the images mentioned in Fig 16. In the first image with the square mask, the output was a little blur at the masked pixels. In the second image, the eyes of both persons were missing. We would like to improve the accuracy of the images obtained by refining the network and training with various masks of different colors, shapes.

#### Conclusion

For the problem statement, we have taken i.e., Image Inpainting using Deep Learning

techniques, we have trained a model using Spectral Normalisation Patch GAN using 

Gated convolution and Self Attention. During the training of this model, Generator loss, 	Discriminator loss, and Reconstruction loss are taken into account. The trained model is

tested using different masks and different resolutions. The results obtained from 	testing with different types of images was satisfactory with good ssim values. 

## TransGAN

#### Dataset and Preprocessing

**Used Dataset: CelebA**

CelebFaces Attributes Dataset (CelebA) is a large-scale face attributes dataset with more than 200K celebrity images, each with 40 attribute annotations. The images in this dataset cover large pose variations and background clutter. Used a total of 11k images and splitted into 10k training samples and 1k testing samples.

#### Model

TransGAN is a GAN model with two Transformer architectures for the generator and discriminator. Convolutions are commonly used in GAN architecture. Convolutions are replaced by transformers in TransGAN.

GAN is based exclusively on Transformer architectures and does not require convolutions. The following are the components of a basic TransGAN:

- A generator that improves feature resolution while simultaneously reducing the size of embeddings;
- A discriminator that only acts on specific portions of an image.

<div class="row justify-content-center">
    <div class="col-sm-8 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/image-inpainting/12.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Overview of framework with TransGAN    
</div>


#### Architecture

**Basic Block:** Similar to transformer encoder, it consists of a multi-head self-attention module and the second part is a feed-forward MLP with GELU non-linearity. It has a residual network with each layer implementing through layer normalization.

**Image as input:** Similar to Vision Transformers, we are dividing the image into 16 \* 16 patches and encoded to the generator

**Memory Friendly Generator:** Similar to GANs, it consists upscaling feature to gradually increase the input sequence and reduce the embedding dimension

**Patch-level Discriminator:** As we need the information only it’s real or fake, the image is divided into 8 \* 8 patches and passed to the transformer encoder, at last, it is flattened and a cls token is attached which confirms if it is real or fake. 


**Generator**

|**Stage**|**Layer**|**Input Shape**|**Output Shape**|
| - | - | - | - |
|-|Image|512|(8 x 8) x 1024|
|<p></p><p></p><p>1</p>|<p>Block</p><p>Block</p><p>Block</p><p>Block</p><p>Block</p>|<p>(8 x 8) x 1024</p><p>(8 x 8) x 1024</p><p>(8 x 8) x 1024</p><p>(8 x 8) x 1024</p><p>(8 x 8) x 1024</p>|<p>(8 x 8) x 1024</p><p>(8 x 8) x 1024</p><p>(8 x 8) x 1024</p><p>(8 x 8) x 1024</p><p>(8 x 8) x 1024</p>|
|<p></p><p></p><p>2</p>|<p>PixelShuffle</p><p>Block</p><p>Block</p><p>Block</p><p>Block</p>|<p>(8 x 8) x 1024</p><p>(16 x 16) x 256</p><p>(16 x 16) x 256</p><p>(16 x 16) x 256</p><p>(16 x 16) x 256</p>|<p>(16 x 16) x 256</p><p>(16 x 16) x 256</p><p>(16 x 16) x 256</p><p>(16 x 16) x 256</p><p>(16 x 16) x 256</p>|
|<p></p><p>3</p>|<p>PixelShuffle</p><p>Block</p><p>Block</p>|<p>(16 x 16) x 256</p><p>(32 x 32) x 64</p><p>(32 x 32) x 64</p>|<p>(32 x 32) x 64</p><p>(32 x 32) x 64</p><p>(32 x 32) x 64</p>|
|-|Linear Layer|(32 x 32) x 64|32 x 32 x 3|

**Discriminator**


|**Stage**|**Layer**|**Input Shape**|**Output Shape**|
| - | - | - | - |
|-|Linear Layer|(32 x 32) x 64|(16 x 16) x 192|
|<p></p><p></p><p>1</p>|<p>Block</p><p>Block</p><p>Block</p><p>AvgPooling</p><p>Concatenate</p>|<p>(16 x 16) x 192</p><p>(16 x 16) x 192</p><p>(16 x 16) x 192</p><p>(16 x 16) x 192</p><p>(8 x 8) x 192</p>|<p>(16 x 16) x 192</p><p>(16 x 16) x 192</p><p>(16 x 16) x 192</p><p>(8 x 8) x 192</p><p>(8 x 8) x 384</p>|
|<p></p><p>2</p>|<p>Block</p><p>Block</p><p>Block</p>|<p>(8 x 8) x 384</p><p>(8 x 8) x 384</p><p>(8 x 8) x 384</p>|<p>(8 x 8) x 384</p><p>(8 x 8) x 384</p><p>(8 x 8) x 384</p>|
|<p></p><p>-</p><p></p>|<p>Add CLS Token</p><p>Block</p><p>CLS Head</p>|<p>(8 x 8) x 384</p><p>(8 x 8 + 1) x 384</p><p>1 x 384</p>|<p>(8 x 8 + 1) x 384</p><p>(8 x 8 + 1) x 384</p><p>1</p>|


#### Training Information

Used 130 epochs for training our model. Used 10K images as training samples and 1K images as testing samples. Captured the loss functions and SSIM values.

**Difficulties:** It took a lot of time to train on our dataset. It needs a lot of GPU memory to train for larger sizes.

**GPU used:** Kaggle's Free GPU



#### Results


<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/image-inpainting/13.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Results of input given with  mask
</div>

#### Limitations


Our model didn’t produce great results for the images mentioned in the above image. Transformers need a lot of GPU memory to train for larger image sizes and for large datasets. 

### Transformer vs CNN

When it comes to NLP models, these transformers have a high success rate, and they're currently being used on images for image recognition tasks. ViT separates the images into visual tokens, whereas CNN uses pixel arrays. The visual transformer splits an image into fixed-size patches, embeds one each appropriately, and sends positional embedding to the transformer encoder as an input. Furthermore, ViT models beat CNNs in terms of computing efficiency and accuracy by nearly four times.

ViT's self-attention layer allows you to embed information globally throughout the entire image. The model also uses training data to represent the measured parallel of image patches in order to reconstruct the image's structure.

The transformer encoder consists of the following components:

**MSP (Multi-Head Self Attention Layer):** This layer concatenates all of the attention outputs to the right dimensions in a linear way. The numerous attention heads in an image assist in the training of local and global dependencies.

**Layer Norm (LN):** This is added before each block as there are no new dependencies between the training images. As a result, the training time and overall performance are improved.

Furthermore, residual connections are included after each block because they allow components to pass directly through the network without having to go through non-linear activations.The MLP layer implements the classification head in the instance of image classification. At pre-training time, it uses one hidden layer and a single linear layer for fine-tuning.

**3.4 Adopting transformers for Inpainting** 

Spectral Norm is used in the discriminator to attain stability in training. Here, a higher learning rate is used for regularised discriminator in order to solve the problem of slow learning than the Generator. L2 normalization with the power iterations [7] is applied for the images in the discriminator part. 

We used Self-attention instead of contextual attention as mentioned in the paper.                                                                                                               We have used this in place of contextual attention for attaining better results as it captures the global knowledge of the image. This self-attention is the last step of each Generator and Discriminator.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/image.gif" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Image Inpainting results
</div>

## Abbreviations

SN - Spectral Normalisation

DNN- Deep Neural Networks 

CNN- Convolutional Neural Networks 

Recon - Reconstruction Loss 

GAN - Generative Adversarial Network 

SSIM - Structural Similarity Index

## References

[1] D. Pathak, P. Krahenbuhl, J. Donahue, T. Darrell, and A. A. Efros. Context encoders: Feature learning by inpainting. In Proc. IEEE Conference on Computer Vision and Pattern Recognition, pages 2536–2544, 2016. 

[2] J. Yu, Z. Lin, J. Yang, X. Shen, X. Lu, and T. S. Huang. Generative image inpainting with contextual attention. In Proc. IEEE Conference on Computer Vision and Pattern Recognition, pages 5505–5514, 2018.

[3] K. Nazeri, E. Ng, T. Joseph, F. Qureshi, and M. Ebrahimi. Edgeconnect: Generative image inpainting with adversarial edge learning. IEEE International Conference on Computer Vision Workshop, 2019.

[4] G. Liu, F. A. Reda, K. J. Shih, T.-C. Wang, A. Tao, and B. Catanzaro. Image inpainting for irregular holes using partial convolutions. In Proc. European Conference on Computer Vision, 2018.

[5] O. Ronneberger, P. Fischer, and T. Brox. U-net: Convolutional networks for biomedical image segmentation. In International Conference on Medical Image Computing and Computer-Assisted Intervention, pages 234–241, 2015.

[6] Yu, Jiahui, et al. "Free-form image inpainting with gated convolution." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2019.

[7] Miyato, Takeru, et al. "Spectral normalization for generative adversarial networks." arXiv preprint arXiv:1802.05957 (2018). 

[8] Deepak Pathak, Philipp Krahenbuhl, Jeff Donahue, Trevor Darrell, and Alexei A Efros. Context encoders: Feature learning by inpainting. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 2536–2544, 2016.

[9]  Jia-Bin Huang, Johannes Kopf, Narendra Ahuja, and Sing Bing Kang. Transformation guided image completion. In Computational Photography (ICCP), 2013 IEEE International Conference on, pages 1–9. IEEE, 2013.

[10] Yann N Dauphin, Angela Fan, Michael Auli, and David Grangier. Language modeling with gated convolutional networks. In Proceedings of the 34th International Conference on Machine Learning-Volume 70, pages 933–941. JMLR. org, 2017.

[11] Raymond A Yeh, Chen Chen, Teck Yian Lim, Alexander G Schwing, Mark Hasegawa-Johnson, and Minh N Do. Semantic image inpainting with deep generative models. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 5485–5493, 2017.

[12] Yuhang Song, Chao Yang, Yeji Shen, Peng Wang, Qin Huang, and C-C Jay Kuo. Spg-net: Segmentation prediction and guidance network for image inpainting. arXiv preprint arXiv:1805.03356, 2018.

[13] Vaswani, Ashish, et al. "Attention is all you need." arXiv preprint arXiv:1706.03762 (2017).

[14] Dosovitskiy, Alexey, et al. "An image is worth 16x16 words: Transformers for image recognition at scale." arXiv preprint arXiv:2010.11929 (2020).

[15] Jiang, Yifan, Shiyu Chang, and Zhangyang Wang. "Transgan: Two transformers can make one strong gan." arXiv preprint arXiv:2102.07074 1.3 (2021).


[ref1]: Aspose.Words.787ad79e-31af-44a5-9b2e-58d0ff25692e.002.png



