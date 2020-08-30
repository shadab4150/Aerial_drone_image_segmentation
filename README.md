# Aerial_drone_image_segmentation

### Dataset Resource: 
* **https://www.tugraz.at/index.php?id=22387**

### Dataset Overview


The Semantic Drone Dataset focuses on semantic understanding of urban scenes for increasing the safety of autonomous drone flight and landing procedures. The imagery depicts more than 20 houses from nadir (bird's eye) view acquired at an altitude of 5 to 30 meters above ground. A high resolution camera was used to acquire images at a size of 6000x4000px (24Mpx). The training set contains 400 publicly available images.

***

## Colab Notebook for Running yourself [**kernel**](https://colab.research.google.com/drive/1BXoew0VpMxWu1a0RKoV18iqz9SkkfnrI?usp=sharing)

#### Aerial Semantic Segmentation Drone Sample Images with mask

***

![kd](https://github.com/shadab4150/Aerial_drone_image_segmentation/blob/master/image_drone/drone1.png)

![kd](https://github.com/shadab4150/Aerial_drone_image_segmentation/blob/master/image_drone/drone5.png)

![kd](https://github.com/shadab4150/Aerial_drone_image_segmentation/blob/master/image_drone/drone4.png)

***


## What is semantic segmentation ?

* Source: **https://divamgupta.com/image-segmentation/2019/06/06/deep-learning-semantic-segmentation-keras.html**

* **Semantic image segmentation is the task of classifying each pixel in an image from a predefined set of classes.**

***

In the following example, different entities are classified.

![kd](https://divamgupta.com/assets/images/posts/imgseg/image15.png?style=centerme)


***


In the above example, the pixels belonging to the bed are classified in the class “bed”, the pixels corresponding to the walls are labeled as “wall”, etc.

In particular, our goal is to take an image of size W x H x 3 and generate a W x H matrix containing the predicted class ID’s corresponding to all the pixels.

***
![kd](https://divamgupta.com/assets/images/posts/imgseg/image14.png?style=centerme)

***

Usually, in an image with various entities, we want to know which pixel belongs to which entity, For example in an outdoor image, we can segment the sky, ground, trees, people, etc.


***

### Data block to feed the model
* Created using fastai datablocks API.
```
data = (SegmentationItemList.from_folder(path=path/'original_images')  # Location from path
        .split_by_rand_pct(0.1)                          # Split for train and validation set
        .label_from_func(get_y_fn, classes=codes)      # Label from a above defined function
        .transform(get_transforms(), size=src, tfm_y=True)   # If you want to apply any image Transform
        .databunch(bs=4)                                   # Batch size  please decrese batch size if cuda out of memory
        .normalize(imagenet_stats))            # Normalise with imagenet stats
data.show_batch(rows=3)
```
***
![kd](https://github.com/shadab4150/Aerial_drone_image_segmentation/blob/master/image_drone/data_block_drone.png)

***
## Model | unet_learner

### Fastai's unet_learner
* Source [**Fast.ai**](www.fast.ai)

* This module builds a dynamic U-Net from any backbone **pretrained on ImageNet**, automatically inferring the intermediate sizes.

![kd](http://deeplearning.net/tutorial/_images/unet.jpg)

* **This is the original U-Net. The difference here is that the left part is a pretrained model.**

* **This U-Net will sit on top of an encoder ( that can be a pretrained model -- eg. resnet50 ) and with a final output of num_classes.**

```
void_code = -1
def accuracy_mask(input, target):
    target = target.squeeze(1)
    mask = target != void_code
    return (input.argmax(dim=1)[mask]==target[mask]).float().mean()

arch =  pretrainedmodels.__dict__["resnet50"](num_classes=1000,pretrained="imagenet")

learn = unet_learner(data, # DatBunch
                     arch, # Backbone pretrained arch
                     metrics = [metrics], # metrics
                     wd = wd, bottle=True, # weight decay
                     model_dir = '/kaggle/working/') # model directory to save
```
### Training :
* Learning rate : Used fastai's lr_find() function to find an optimal learning rate.

```
learn.lr_find()
learn.recoder.plot()
```
![kd](https://github.com/shadab4150/Aerial_drone_image_segmentation/blob/master/image_drone/lr_finder.png)

```
callbacks = SaveModelCallback(learn, monitor = 'accuracy_mask', every = 'improvement', name = 'best_model' )
learn.fit_one_cycle(10, slice(lr), pct_start = 0.8, callbacks = [callbacks] )
```
***

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>drone_accuracy_mask</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>2.106114</td>
      <td>1.835669</td>
      <td>0.483686</td>
      <td>00:16</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1.697477</td>
      <td>1.331731</td>
      <td>0.618320</td>
      <td>00:12</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1.462504</td>
      <td>1.227808</td>
      <td>0.659259</td>
      <td>00:13</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1.455247</td>
      <td>1.147135</td>
      <td>0.691440</td>
      <td>00:12</td>
    </tr>
    <tr>
      <td>4</td>
      <td>1.336305</td>
      <td>1.115214</td>
      <td>0.691359</td>
      <td>00:12</td>
    </tr>
    <tr>
      <td>5</td>
      <td>1.247270</td>
      <td>1.168395</td>
      <td>0.669359</td>
      <td>00:12</td>
    </tr>
    <tr>
      <td>6</td>
      <td>1.310875</td>
      <td>1.181834</td>
      <td>0.672961</td>
      <td>00:13</td>
    </tr>
    <tr>
      <td>7</td>
      <td>1.196860</td>
      <td>1.115905</td>
      <td>0.708513</td>
      <td>00:12</td>
    </tr>
    <tr>
      <td>8</td>
      <td>1.131353</td>
      <td>0.888599</td>
      <td>0.778729</td>
      <td>00:12</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.983422</td>
      <td>0.799664</td>
      <td>0.851884</td>
      <td>00:12</td>
    </tr>
     <tr>
      <td>10</td>
      <td>0.973422</td>
      <td>0.779664</td>
      <td>0.881884</td>
      <td>00:12</td>
    </tr>
    <tr>
      <td>11</td>
      <td>0.953422</td>
      <td>0.749664</td>
      <td>0.911884</td>
      <td>00:12</td>
    </tr>
  </tbody>
</table>

***

### Results |

Intial dynamic unet on top of an encoder ( resnet34 pretrained = 'imagenet' ), trained for 10 epochs gave an accuracy of **91.1%** .
```
learn.show_results(rows=3, figsize=(12,16))

```

![kd](https://github.com/shadab4150/Aerial_drone_image_segmentation/blob/master/image_drone/results_drone.png)
