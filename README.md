# Styled_vae_vq
<div align="center">
Author: lidehu 2201210265@stu.pku.edu.cn
    在soul app 实习期间的工作
</div>

## 动机： 
- 业务需求:需要根据聊天上下文,自发的生成图片,比如:聊天时候,a:我去过黄山,秋天去的,可漂亮了.b:巧了,我冬天去的,我还滑雪了呢,(发送对应生成的照片),我之前在长白山滑过雪(发送生成的照片,此时注意人物画像要一致),比黄山还要好玩些呢.a:哇,冬天的黄山真好看,还有云海哎(根据生成的图片做理解).所以我选取了自回归式生成与理解方案,完全的端到端多模态大模型.
- 注意:此方案是用来总结图像的,细节方面是有欠缺的,对于生成而言,可以理解为它提供了很详细的'图像语言'描述,很多时候,人类是很难按照某种固定规律还不能太过琐碎方式去描述图像的,而且还需要很容易根据简单的语言文字描述去生成图像描述,详情见重建部分说明.
- 与titok出发点不同,所以网络结构是不一样的,最初的解码器直接是扩大俩倍的stylegan2,所以这份工作起名style_vae_vq,与titok效果也不同.此份工作在5月底展开,由于组内图像基础相当薄弱(正在转向),万事靠自己摸索,加上还有其他任务,所以进展缓慢.
- 理解:目前视觉encoder输出token数量多,没有规律,llm关注重点偏移,出现幻觉;通俗来说,说话没有重点,没有规则逻辑.(经过实验:titok也未观察出规律)
- 生成:数据不对齐(输入输出为一对n的映射关系),加上按照一维顺序预测下一个token,token联系不够紧密,下一个token预测,映射关系依然很发散,概率分布比较平均,预测难度较大.(经实验普通的vqvae自回归生图方案,前面给足原图的多尺度小图token,是很容易生成原图的token的,但是直接生成小图token,存在多样性问题)
- 自监督通用视觉模型:目前方案存在粗粒度和细粒度不能很好兼容问题.还原图片/特征为代表的自监督方案细粒度理解能力强,但是整体理解偏弱,伪标签做有监督的训练方案整体理解能力强,但是细节处理偏弱.
- 解决方案：提出图像语言:Styled_vae_vq,将图像按序离散成36个(根据压缩比确定)从高级到低级的属性token,编码器提供易对齐的有固定规则的序列,提供更优的输入输出映射关系.使用时候和其他视觉编码器拼接.
- 插播:有愿意一起写论文的朋友吗,共一或者你通讯,本人写作水平有限. 
- 旧版本:详情请见https://github.com/hamigualisingl/Styled_vae_vq/blob/main/READMEV2.md

## 关键部分代码：
- 编码器部分:
    ```
    for r in self.resblocks_exper:
        x = checkpoint(r, x, attn_mask)#最后6层使用了专家,每层37个专家,一个属性一个专家
    mu=self.progject_mean(x[257:]) #降维度,制造信息瓶颈,第二阶段需要量化的值也是这个
    ################################################################################
    mu_flattened = mu.view(-1, 128)#其中 self.V2 = nn.Parameter(scale * torch.randn(self.emb_dim,self.emb_dim))#可以看做一个连接层
    similarity = cosine_similarity(mu_flattened.float().unsqueeze(1), self.V2.float(), dim=2)+1
    similarity= similarity/torch.sum(similarity, dim=-1, keepdim=True)#线性加权，未使用softmax暂时没想好温控策略,这边也可以减少误差的影响,但是操作不当会增加误差
    weighted_sum = self.ln_cosin(torch.matmul(similarity, self.V2))
    output = weighted_sum.view(mu.shape)
     ###############这边是为了减轻第二阶段量化影响的, 保证值连续含义是连续的,这样存在误差也无妨,本来就是总结图像，不指望他还原,目前追求还原质量是担心链路太长,每个环节都差一些,不好找原因
    ```
- 解码器部分-条件添加方式如下:
    ```
    for index, r in enumerate(self.condtionTransformer):
        x = r(x,conditon[index*2:(index+1)*2], index)#每层添加俩个条件
    ```
    ```
    norm_hidden_states = self.norm1(x,self.ln_1(condation[0]))
    x=torch.cat([torch.zeros(2, *x.shape[1:], device=x.device, dtype=x.dtype),x], dim=0)##条件先norm，然后参与交互
    norm_hidden_states=torch.cat([self.ln_11(condation[0:1]),self.ln_22(condation[1:2]),norm_hidden_states], dim=0)#条件先交互后norm，
    x = x + self.attn1(norm_hidden_states, norm_hidden_states, norm_hidden_states, need_weights=False, attn_mask=attn_mask)[0]#([516, 516]) torch.float32
    x = x + self.mlp(self.norm2(x,self.ln_2(condation[1])))
    ```

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

    数据集: YFCC15M(随机挑选7.3M)+3.3M(混杂数据集)(最初实验在59w数据规模训练,对应噪音比例也增大,在YFCC15M测试,重建效果挺不错,所以最终版本只在1060w数据规模训练,降低噪音比例).超参数:lr 1e-4;12epoch(资源限制);bs 16*256;optim.AdamW,lr_scheduler "cosine".有多个版本,解码器有俩个版本(768/512),编码器有俩个版本(输出128/64).[Google Drive](https://drive.google.com/file/d/1IO_RgGXrjLhUlhmRHwCmeIoNVH6jMGab/view?usp=sharing)(版本:768,128)

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


