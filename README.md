# Styled_vae_vq
<div align="center">
Author: lidehu 2201210265@stu.pku.edu.cn
    在soul app 实习期间的工作
</div>

## 动机： 
应对图像理解与自回归生成难题,比如:理解-注意力分散,出现幻觉;生成-数据不对齐,下一个token概率分布比较平均,自回归生成困难,提出Styled_vae_vq,将图像按序离散成36个从高级到低级的属性token,力求编码器提供了合理的易对齐的序列,而非训出有强大拟和能力的解码器.使用时候可以和其他vqvae拼接.

## 模型结构
- 编码器参数远大于解码器参数.
- 编码器:1024维度.一个输入卷积层input_cov,256个位置token和36个特殊token,24层Transformer(18dense+6experts),末尾为投影层,输出128维度.
- 解码器:768维度.256个位置token和36个占位token,18个condtionTransformer层,6层Transformer,输出卷积层out_cov.
## 数据流动流程：
- 编码器:img(bs,3,256,256)->input_cov(img)->(256,bs,1024)->add(256个位置token).
->cat(x,36个特殊token)->Transformer->取出后36个token降维度作为condtion.
- 解码器:256个位置token->condtionTransformer->Transformer->取后256个token->out_cov->重建损失,感知损失.
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
- 条件添加方案解释:整体添加方式类似于stylegan2的w+,相关stylegan2W+的分析不再赘述.为了进一步加强生成序列的纠缠关联有逻辑,和促使36个token,是从高到低的属性组合,使用了占位token,条件token需要与对应的占位token相加,而占位token到达本层时候,已经被前面的条件token改变,影响.因此如果36个token,是从低到高的属性组合,或者没有相关性,解码器复原图片变得很困难,浓浓的眉毛属性影响到性别是男性属性,这个会给复原带来很大难度,但如果先指明这个人是男性属性然后影响眉毛属性,这就合理些.
## 训练流程
- 由于任务比较困难,采取俩阶段训练策略.未采用fsq之类的操作原因如下:我们是要得到一个合理的序列或者特征,不能是依靠解码器拟合回去(需要训很多epoch).
- 阶段一:编码器输出连续值,添加噪音后送入解码器
- 阶段二:通过k-means聚类,得到词表,编码器(固定)输出连续值->量化
- ### Environment installation

    ```
    pip install -r requirments.txt
    ```


- ### Pretrained Model Weight

    数据集:主要是在cc3m随机抽选的56万张图片训,然后在imagenet等其他数据进一步训练.很快放出来,目前第一阶段(56万张图片)已经训练完成,训练数据外的图片重建效果不错(目前模型大小为8.9G,主要是编码器较大),后面会在laion400m进一步训练.

- ### Training

    Start training by run
    ```
    bash /mnt/data/user/lidehu/vae/ALIP/run_zeroshot.sh 64 1e-4 /mnt/data/user/lidehu/vae/ALIP/out_put_stage1_6expert_std_noise_1_pect_1  1024 200#注意数据集路径更换!
    ```

- ### Use

  

    ```
    python  how_to_use.py#注意图片路径和模型路径更换!
    ```
   



## Acknowledgement

感谢soul app自然语言算法组的支持,宽松的研究环境,富裕的计算资源.

## License

.



