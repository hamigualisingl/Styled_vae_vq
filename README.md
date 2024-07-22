# Styled_vae_vq
<div align="center">
Author: lidehu 2201210265@stu.pku.edu.cn
    在soul app 实习期间的工作
</div>

## 动机： 
- 理解:目前视觉encoder输出token数量多,地位平等,注意力分散,关注重点偏移,出现幻觉;通俗来说,说话没有重点,没有规则逻辑.
- 生成:数据不对齐,加上按照一维顺序预测下一个token,破坏了图像特性,token联系不够紧密,导致大部分情况下 下一个token预测,约束不足,概率分布比较平均,预测难度较大.
- 解决方案：Styled_vae_vq,将图像按序离散成36个从高级到低级的属性token,编码器提供易对齐的有固定规则且联系紧密的序列和用于初始化llm图像词表的特征矩阵(充分利用视觉编码器信息,通过特征提供了序列之间的关系,优于仅提供序列的做法).使用时候可以和其他vqvae拼接.

## 模型结构
- 编码器参数远大于解码器参数.
- 编码器:1024维度.一个输入卷积层input_cov,256个位置token和36个特殊token,24层Transformer(18dense+6experts(每层37个专家)),末尾为投影层,输出128/64维度.
- 解码器:768/512维度.256个位置token和36个占位token,18个condtionTransformer层,6层Transformer,输出卷积层out_cov.
## 数据流动流程：
- 编码器:img(bs,3,256,256)->input_cov(img)->(256,bs,1024)->add(256个位置token).
->cat(x,36个特殊token)->Transformer->取出后36个token降维度至128/64(制造信息瓶颈)作为condtion.
- 解码器:256个位置token+36个占位token->condtionTransformer->Transformer->取后256个token->out_cov->重建损失,感知损失.
- 条件添加方式如下:
    ```
    for index, r in enumerate(self.condtionTransformer):
        x = r(x,conditon[index*2:(index+1)*2], index)#每层添加俩个条件
    ```
    ```
    zeros_like_x = torch.zeros_like(x)#(36+256,bs,dim)
    zeros_like_x[index*2:(index+1)*2]=zeros_like_x[index*2:(index+1)*2]+ condation  #条件和占位token相加
    x=x+ zeros_like_x  #条件和占位token相加,注意此时的占位token在前向过程中已经被之前的条件token影响.
    norm_hidden_states = self.norm1(x,self.ln_1(self.condation_1(x[index*2])+condation[0]))#自适应归一化。条件token发挥作用前需要和占位token交互,这样后面条件token起什么作用受到前一个token牵扯
    x = x + self.attn1(norm_hidden_states, norm_hidden_states, norm_hidden_states, need_weights=False, attn_mask=attn_mask)[0]#([516, 516]) torch.float32
    x = x + self.mlp(self.norm2(x,self.ln_2(self.condation_2(x[index*2+1])+condation[1])))
       
    ```
- 条件添加方案解释:整体添加方式类似于stylegan2的w+,相关stylegan2W+的分析不再赘述.为了进一步加强生成序列的纠缠关联有逻辑,和促使36个token,是从高到低的属性组合,使用了占位token.条件token需要与对应的占位token相加,而占位token到达本层时候,已经被前面的条件token改变,影响.因此如果36个token,是从低到高的属性组合,或者没有相关性,解码器复原图片变得很困难.比如:浓浓的眉毛属性影响到性别是男性属性,这个会给复原带来很大难度,但如果先指明这个人是男性属性然后影响眉毛属性,这就合理些.
## 训练流程
- 由于任务比较困难,采取俩阶段训练策略.未采用fsq之类的操作原因如下:fsq等方案存在序列跳转困难问题,我们是要通过编码器得到一个合理的序列和特征,不能是依靠训一个强大的解码器拟合回去(需要训很多epoch).
- 阶段一:编码器输出连续值,添加噪音后送入解码器.参数最初的vae,可以适当添加kl_loss.
- 阶段二:通过k-means聚类,得到词表(n,128/64).img->编码器(固定)输出连续值->量化(固定)->升维度(参与训练,维度为llm词表维度一致,用于llm图像词表初始化)->降维度至768/512(参与训练,注意!此时去掉阶段一段128/64->768/512层)->解码器(固定).
- 阶段三:训练llm,预期方案(最近忙着秋招,之后会单起一个项目介绍):使用俩个预测头和俩套词表.先训练新增参数固定其他,进行图像到文字的对齐,后面统一打开.llm预期利用mata变色龙模型初始化.
- 特别的生成部分:styled_vae_vq部分串行预测,后面的vqvae并行预测
- ### Environment installation

    ```
    pip install -r requirments.txt
    ```


- ### Pretrained Model Weight

    数据集: YFCC15M(随机挑选7.3M)+3.3M(混杂数据集)(最初实验在59w数据规模训练,对应噪音比例也增大,在YFCC15M测试,重建效果挺不错,所以最终版本只在1060w数据规模训练,降低噪音比例).超参数:lr 1e-4;9epoch;bs 16*256;optim.AdamW,lr_scheduler "cosine".有多个版本,解码器有俩个版本(768/512),编码器有俩个版本(输出128/64).
  [Google Drive](https://drive.google.com/file/d/1AqSHisCKZOZ16Q3sYguK6zIZIuwwEriE/view?usp=share_link)(版本:768,128)

- ### Training

    Start training by run
    ```
    bash Styled_vae_vq/run.sh 64 1e-4 /mnt/data/user/lidehu/vae/ALIP/out_put_stage1_6expert_std_noise_1_pect_1  1024 200#注意数据集路径更换!
    ```

- ### Use

  

    ```
    python  how_to_use.py#注意图片路径和模型路径更换!
    ```
   



## Acknowledgement

感谢soul app自然语言算法组的支持,宽松的研究环境,富裕的计算资源.

## 重建效果 
- 注意!没有添加对抗损失.因为图像的特性,模型要把位置对应的vqvae方案(改变其中一个token,解码后对应位置内容也会改变)作为最后一公里,提供了几乎一对一的映射关系后由普通vqvae去还原.

.



