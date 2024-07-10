# Styled_vae_vq
<div align="center">
Author: lidehu
</div>

## 动机： 
应对图像理解与自回归生成难题，比如:理解-注意力分散，出现幻觉；生成-数据不对齐，下一个token选择过多，自回归生成困难，提出Styled_vae_vq，将图像按序离散成36个从高级到低级的属性token。使用时候可以和其他vqvae拼接。


## 模型结构
- 编码器：一个输入卷积层input_cov，256个位置token和36个特殊token，24层Transformer,末尾为投影层预测头。
- 解码器：256个位置token和36个占位token，18个condtionTransformer层，6层Transformer，输出卷积层out_cov。
## 数据流动流程：
- 编码器：img(bs,3,256,256)->input_cov(img)->(256,bs,1024)->add(256个位置token)
->cat(x,36个特殊token)->Transformer->取出后36个token降维度作为condtion
- 解码器：256个位置token->condtionTransformer->Transformer->取后256个token->out_cov->重建损失
- 条件添加方式如下:
    ```
    for index, r in enumerate(self.condtionTransformer):
        x = r(x,conditon[index*2:(index+1)*2], index)
    ```
    ```
    zeros_like_x = torch.zeros_like(x)#(36+256,bs,dim)
    zeros_like_x[index*2:(index+1)*2]=zeros_like_x[index*2:(index+1)*2]+ condation
    x=x+ zeros_like_x
    norm_hidden_states = self.norm1(x,self.ln_1(self.condation_1(x[index*2])+condation[0]))
    x = x + self.attn1(norm_hidden_states, norm_hidden_states, norm_hidden_states, need_weights=False, attn_mask=attn_mask)[0]#([516, 516]) torch.float32
    x = x + self.mlp(self.norm2(x,self.ln_2(self.condation_2(x[index*2+1])+condation[1])))
       
    ```
- 条件添加方案解释：整体添加方式类似于stylegan2的w+，相关stylegan2W+的分析不再赘述。为了进一步加强生成序列的纠缠关联有逻辑，和促使36个token，是从高到低的属性组合，使用了占位token，当前条件token2需要与占位token相加，而占位token到达本层时候，已经被前面的条件token改变，影响。因此如果36个token，是从低到高的属性组合，或者没有相关性，解码器复原图片变得很困难，浓浓的眉毛属性影响到性别是男性属性，这个会给复原带来很大难度，但如果先指明这个人是男性属性然后影响眉毛属性，这就合理些。
## 训练流程
- 由于任务比较困难,采取俩阶段训练策略
- 阶段一:编码器输出连续值，添加噪音后送入解码器
- 阶段二:通过k-means聚类，得到词表，编码器(固定)输出连续值->量化
- ### Environment installation

    ```
    pip install -r requirments.txt
    ```
- ### 
    
    1、Download YFCC15M

    The YFCC15M dataset we used is [YFCC15M-DeCLIP](https://arxiv.org/abs/2110.05208), we download it from the [repo](https://github.com/AdamRain/YFCC15M_downloader), finally we successful donwload 15061515 image-text pairs.

    2、Generate synthetic caption

    In our paper, we use OFA model to generate synthetic captions. You can download model weight and scripts from the [OFA](https://github.com/OFA-Sys/OFA) project.

    3、Generate rec files

    To improve the training efficience, we use [MXNet](https://github.com/apache/mxnet) to save the YFCC15M dataset to rec file, and use NVIDIA [DALI](https://github.com/NVIDIA/DALI) to accelerate data loading and pre-processing. The sample code to generate rec files is in [data2rec.py](data2rec.py).

- ### Pretrained Model Weight

    You can download the pretrained model weight from [Google Drive](https://drive.google.com/file/d/1AqSHisCKZOZ16Q3sYguK6zIZIuwwEriE/view?usp=share_link) or [BaiduYun](https://pan.baidu.com/s/10dFfvGMWeaTXUyrZlZlCEw?pwd=xftg), and you can find the traning log in [Google Drive](https://drive.google.com/file/d/1I8gdSQCJAfFamDcVztwW8EQIc_OOK8Xh/view?usp=share_link) or [BaiduYun](https://pan.baidu.com/s/1oz0UVzX2N0Sri7MfwR-kog?pwd=7ki7)

- ### Training

    Start training by run
    ```
    bash scripts/train_yfcc15m_B32_ALIP.sh
    ```

- ### Evaluation

    Evaluate zero shot cross-modal retireval

    ```
    bash run_retrieval.sh
    ```
    Evaluate zero shot classification

    ```
    bash run_zeroshot.sh
    ```



## Acknowledgement

This project is based on [open_clip](https://github.com/mlfoundations/open_clip) and [OFA](https://github.com/OFA-Sys/OFA), thanks for their works.

## License

This project is released under the MIT license. Please see the [LICENSE](LICENSE) file for more information.



