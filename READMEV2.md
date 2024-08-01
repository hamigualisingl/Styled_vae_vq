# Styled_vae_vq
<div align="center">
Author: lidehu 2201210265@stu.pku.edu.cn
    在soul app 实习期间的工作
</div>

## 变动部分：
- 编码器输出特征部分:
    ```
        mu=self.progject_mean(x[257:]) #这边才是要量化的！！量化使用余弦距离
        ###############V2,这边是为了第二阶段减轻量化损失影响的！
        mu_flattened = mu.view(-1, 128)
        similarity = cosine_similarity(mu_flattened.float().unsqueeze(1), self.V2.float(), dim=2)
        similarity= similarity/torch.sum(similarity, dim=-1, keepdim=True)#线性加权，未使用softmax暂时没想好温控策略,这边也可以减少误差的影响,但是操作不当会增加
        weighted_sum = self.ln_cosin(torch.matmul(similarity, self.V2))
        output = weighted_sum.view(mu.shape)
        ###############这边是为了减轻量化影响的, 保证含义是连续的,这样存在误差也无妨,本来就是总结图像，不指望他还原,目前追求还原质量是担心链路太长,每个环节都差一些,不好找原因
        return     output
    ```
- 解码器部分:去掉了占位token,条件只做自适应归一化,预测目标为vqgan的特征,使用交叉熵,不直接回归像素
- 变动解释:给第二阶段添加容错机制. 新版本训练中.
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
