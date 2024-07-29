# Styled_vae_vq
<div align="center">
Author: lidehu 2201210265@stu.pku.edu.cn
    在soul app 实习期间的工作
</div>

## 动机： 
- 业务需求:需要根据聊天上下文,自发的生成图片,比如:聊天时候,a:我去过黄山,秋天去的,可漂亮了.b:巧了,我冬天去的,我还滑雪了呢,(发送对应生成的照片),我之前在长白山滑过雪(发送生成的照片,此时注意人物画像要一致),比黄山还要好玩些呢.a:哇,冬天的黄山真好看,还有云海哎(根据生成的图片做理解).所以我选取了自回归式生成与理解方案,完全的端到端多模态大模型.
- 注意:此方案是用来总结图像的,细节方面是有欠缺的,对于生成而言,可以理解为它提供了很详细的'图像语言'描述,很多时候,人类是很难按照某种固定规律还不能太过琐碎方式去描述图像的,而且还需要很容易根据简单的语言文字描述去生成图像描述,详情见重建部分说明.
- 与titok出发点不同,所以网络结构是不一样的,最初的解码器直接是扩大俩倍的stylegan2,所以这份工作起名style_vae_vq,此份工作在5月底展开,由于组内图像基础相当薄弱(正在转向),万事靠自己摸索,加上还有其他任务,所以进展缓慢.
- 理解:目前视觉encoder输出token数量多,地位平等,注意力分散,关注重点偏移,出现幻觉;通俗来说,说话没有重点,没有规则逻辑.
- 生成:数据不对齐,加上按照一维顺序预测下一个token,破坏了图像特性,token联系不够紧密,导致大部分情况下 下一个token预测,约束不足,概率分布比较平均,预测难度较大.(经实验普通的vqvae自回归生图方案,前面给足原图的多尺度小图token,是很容易生成原图的token的,但是直接生成小图token,存在多样性问题)
- 自监督通用视觉模型:目前方案存在粗粒度和细粒度不能很好兼容问题.还原图片/特征为代表的自监督方案细粒度理解能力强,但是整体理解偏弱,伪标签做有监督的训练方案整体理解能力强,但是细节处理偏弱.
- 解决方案：Styled_vae_vq,将图像按序离散成36个(根据压缩比确定)从高级到低级的属性token,编码器提供易对齐的有固定规则且联系紧密的序列和用于初始化llm图像词表的特征矩阵(充分利用视觉编码器信息,通过特征提供了序列之间的关系,优于仅提供序列的做法).使用时候和其他视觉编码器拼接.
- 插播:有愿意一起写论文的朋友吗,共一或者你通讯,我负责实验,(秋招太忙了,一个人干不来)    
- 最新进展:第二阶段离散时候出现问题,分析如下:离散后特征出现误差.直接回归像素,缓和余地太小,解决方向-添加容错机制:比如考虑借用MaskGIT-VQ的成果,交叉熵方式预测它的encoder的结果,然后借用它的解码器,考虑去掉占位token,避免前面token在重建时候,比重过大.不过之前的版本依然会在理解与生成测试,因为是对图像总结,提供尽可能一对一映射关系,会有细节编码器去生图,不要求自身生成效果.

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
- 条件添加方案解释:整体添加方式类似于stylegan2的w+,相关stylegan2W+的分析不再赘述.为了进一步加强生成序列的纠缠关联有逻辑,和促使36个token,是从高到低的属性组合,使用了占位token.条件token需要与对应的占位token相加,而占位token到达本层时候,已经被前面的条件token改变,影响.因此如果36个token,是从低到高的属性组合,或者没有相关性,解码器复原图片变得很困难.比如:浓浓的眉毛属性影响到性别是男性,这个会给复原带来很大难度,但如果先指明这个人是男性然后影响眉毛属性,这就合理些.
## 训练流程
- 由于任务比较困难,采取俩阶段训练策略,先获取合理的特征,然后量化,合理的特征会增强量化后序列间的逻辑.避免vqvae量化序列跳转困难(fsq更加突出),词表崩塌等问题-我们是要通过编码器得到一个合理的序列和特征,不能是依靠训一个强大的解码器拟合回去.
- 阶段一:编码器输出连续值,添加噪音后送入解码器.参考最初的vae,可以适当添加kl_loss.#训练结束,效果参见重建部分.此部分已经训练结束.
- 阶段二:通过k-means聚类,得到词表(n,128/64).打开解码器编码器,固定词表,暂时不使用一致性损失,只允许序列跳转.目前还在实验阶段,最终方案还未确立.最后会做统一说明,最终采取了那种方案.
- 阶段三:训练llm,预期方案(最近忙着秋招,之后会单起一个项目介绍):使用俩个预测头和俩套词表.先训练新增参数固定其他,进行图像到文字的对齐,后面统一打开.llm预期利用mata变色龙模型初始化.
- 其中生成部分:styled_vae_vq(离散)部分串行预测,后面的vqvae并行预测
- 理解部分:使用连续值/连续
- ### Environment installation

    ```
    pip install -r requirments.txt
    ```


- ### Pretrained Model Weight

    数据集: YFCC15M(随机挑选7.3M)+3.3M(混杂数据集)(最初实验在59w数据规模训练,对应噪音比例也增大,在YFCC15M测试,重建效果挺不错,所以最终版本只在1060w数据规模训练,降低噪音比例).超参数:lr 1e-4;12epoch(资源限制);bs 16*256;optim.AdamW,lr_scheduler "cosine".有多个版本,解码器有俩个版本(768/512),编码器有俩个版本(输出128/64).(这个链接不是权重,目前组内带宽已满无法下载模型,稍等)
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

感谢soul app自然语言算法组.

## 重建效果(连续值重建/离散阶段还在训练中) 
- 注意!没有添加对抗损失,模型也没有得到充分训练.因为图像的特性,模型要把位置对应的vqvae方案(改变其中一个token,解码后对应位置内容也会改变)作为最后一公里,提供了几乎一对一的映射关系后由普通vqvae去还原.
- 分析:非自然图像,属性提取困难,还原有难度-河马(sd生成的图像),力扣界面.
---

### 图像示例

| 原始图像 | 重建图像 | 原始图像 | 重建图像 |
| --- | --- | --- | --- |
| ![原始图像 1](https://github.com/user-attachments/assets/ef0ac4fc-7e4a-4c76-b05b-fb0988a67621) | ![重建图像 1](https://github.com/user-attachments/assets/e0f874aa-ae3d-4bb4-820f-02464f2b0572) | ![原始图像 2](https://github.com/user-attachments/assets/7d4954b5-56ae-4376-8b43-af5f9bdb8bf0) | ![重建图像 2](https://github.com/user-attachments/assets/a9667b03-98f7-451d-ba93-873d129dc7d6) |
| ![原始图像 3](https://github.com/user-attachments/assets/a84b7f6a-98ae-4e1b-a9da-693ad89cfa9c) | ![重建图像 3](https://github.com/user-attachments/assets/e93b503d-57e6-484a-a156-2ac2172e7d58) | ![原始图像 4](https://github.com/user-attachments/assets/f299a918-2f28-4a4e-a9ea-ee5ae8e2f6ad) | ![重建图像 4](https://github.com/user-attachments/assets/588f07f5-448a-4a6c-a56b-05f11fded407)|| ![原始图像 5](https://github.com/user-attachments/assets/dba588a9-59ec-4874-9bea-2b0f86b8fbbb) | ![重建图像 5](https://github.com/user-attachments/assets/cfe7b6df-81e4-470a-9726-0539ba1a0eee) | ![原始图像 6](https://github.com/user-attachments/assets/07f77089-b755-43b3-ae73-03aa1fc42602) | ![重建图像 6](https://github.com/user-attachments/assets/2b6619a3-e673-432c-934e-1ed542ae973a) |
| ![原始图像 7](https://github.com/user-attachments/assets/adf3171d-3138-4ca5-936a-7c0f2a7f37fe)| ![重建图像 7](https://github.com/user-attachments/assets/265a8276-9b54-45b7-b3b4-4089136df7a6)| ![原始图像 8](https://github.com/user-attachments/assets/b5171758-0bd0-46cf-b961-df5442b864da)| ![重建图像 8](https://github.com/user-attachments/assets/98285ac0-b76f-403c-b1ce-c72f377d27cd)|

---


