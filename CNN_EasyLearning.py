import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

#ctrl+鼠标左键，进入类/函数（可进入多层，查看注释）；alt+←，退出类/函数

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

num_epochs= 5  #训练5遍
num_classes =10 #目标类别
batch_size = 100 #训练批次中，每个批次要加载的样本数量（默认值:)1
learning_rate = 0.001

train_dataset = torchvision.datasets.MNIST(root = '../../data/',
                                            train= True,
                                            transform = transforms.ToTensor(),
                                            download=False)
'''
    1、文件存储路径表示方法：
    2、已封装数据集包括：CIFAR-10，以及ImageNet、COCO、MNIST、LSUN
        所有数据集都具有几乎相似的接口（API，application programming interface）
        MNIST（root =存储路径，
            train =是否属于训练集，
            download =是否下载，
            transform =转换格式（一个函数/转换，它接收PIL图像并返回转换后的版本。例如，transforms.RandomCrop）
            target_transform=是否对label转换 （接收目标并对其进行转换的函数/转换）
    
    4、transforms：参见https://pytorch.org/docs/stable/torchvision/transforms.html
        torchvision.transforms.Compose[...]  一起组合几个变换
        torchvision.transforms.CenterCrop(尺寸大小)  将给定的PIL图像裁剪为中心
        torchvision.transforms.ColorJitter（亮度= 0，对比度= 0，饱和度= 0，色调= 0 ）更改图像亮度
        torchvision.transforms.Grayscale（num_output_channels = 1 ）转换灰度图像，默认为1；为3时，则输出RGB
        torchvision.transforms.Pad（padding，fill = 0，padding_mode ='constant' ）：
            padding（int或tuple） - 边框填充。长度为1，则用于4条边；长度为2，则分别填充左/右和上/下；长度为4，分别填充左，上，右和下边框。
            fill（int或tuple） - 常量填充的像素填充值。默认值为0.如果长度为3的元组，则分别用于填充R，G，B通道。仅当padding_mode为常量时才使用此值
            padding_mode（str） -填充类型。应该是：恒定constant, 边缘edge，反射reflect 或对称symmetric。默认值是常量。
    
        transforms成员函数非常多：

        RandomAffine：随机仿射变换
        RandomApply：随机应用给定概率的变换列表
        RandomChoice：应用从列表中随机挑选的单个转换
        RandomCrop：在随机位置裁剪
        RandomGrayscale：随机转换为灰度
        RandomHorizontalFlip：随机水平翻转
        RandomVerticalFlip：随机垂直翻转
        RandomOrder：以随机顺序应用转换列表
        RandomPerspective：随机透视变换
        RandomResizedCrop：随机裁剪为固定大小
        RandomRotation：随机旋转

        Resize：重新调整大小
        LinearTransformation：线性变换
        Normalize（mean，std，inplace = False ）：给定均值、标准差归一化

        ToPILImage：将张量或ndarray转换为PIL图像
        ToTensor：图像转换为张量
'''

test_dataset = torchvision.datasets.MNIST(root = '../../data/',
                                            train = False,
                                            transform = transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                            batch_size = batch_size,
                                            shuffle = True)
'''
    Dataloader(
        dataset = Dataset- 从中​​加载数据的数据集。
        batch_size 
        shuffle = 是否打乱顺序，默认为否
        sampler = 定义从数据集中绘制样本的策略。如果指定，则shuffle必须为False。
        batch_sampler = 与sampler一样，但一次返回一批索引。互斥有batch_size， shuffle，sampler，和drop_last。
        num_workers（int，optional） - 用于数据加载的子进程数。0表示数据将加载到主进程中。（默认值：0）
        collat​​e_fn（callable ，optional） - 合并样本列表以形成小批量。
        pin_memory（bool，optional） - 如果True，数据加载器会在返回之前将张量复制到CUDA固定内存中。
        drop_last（bool，optional） - 当数据集不能被批处理整除时，是否丢弃最后一批，（默认为False，即最后一批数据少）
        timeout（数字，可选） - 如果为正，则为从工作人员收集批处理的超时值。应始终是非负面的。（默认值：0）
        worker_init_fn（callable ，optional） - 如果没有None，则在播种后和数据加载之前，将在每个worker子进程上调用 this，并将worker id（int in ）作为输入。（默认值：）[0, num_workers - 1]None）
'''


test_loader = torch.utils.data.Dataloader(dataset=test_dataset,
                                            batch_size=batch_size,
                                            shuffle = False)

class ConvNet(nn.Module):
    #torch.nn.Module 所有神经网络模块的基类
    def __init__(self,num_classes=10):
        super(ConvNet,self).__init__()
        #继承基类的构造函数，固定写法：super(NewModel, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1,16,kernel_size=5,stride=1,padding=2),  #输入维度1x28x28
            nn.BatchNorm2d(16),  #
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16,32,kernel_size = 5,stride=1,padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2,stride=2)
        )
        self.fc = nn.Linear(7*7*32,num_classes)

    def forward(self,x):
        out =self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size[0],-1)
        out = self.fc(out)
        return out
'''
    1、神经网络类定义，构造函数+前向传播函数

    2、torch.nn.Conv2d（in_channels， 输入通道：灰度图像为1，彩色RGB为3
                    out_channels，   输出通道：与卷积核数量一致
                    kernel_size，   卷积核尺寸，可以为int或者tuple
                    stride = 1，  卷积步长
                    padding = 0， 填充
                    dilation = 1， 
                    groups = 1，
                    bias = True， 偏置
                    padding_mode ='zeros' ）

                    参数kernel_size，stride，padding，dilation可以是：
                    单个int，作用于高度和宽度；tuple，第一int用于高度（相当于几行），第二个同于宽度（相当于几列）
    
    3、torch.nn.Sequential一个连续的容器。模块将按照它们在构造函数中传递的顺序添加到它中
        两种写法：
        顺序传入：
        model = nn.Sequential(
                nn.Conv2d(1,20,5),
                nn.ReLU(),
                nn.Conv2d(20,64,5),
                nn.ReLU()
                )
        有序字典（层数较多时，给每一层起一个名字更明了）：
        model = nn.Sequential(OrderedDict([
                ('conv1', nn.Conv2d(1,20,5)),
                ('relu1', nn.ReLU()),
                ('conv2', nn.Conv2d(20,64,5)),
                ('relu2', nn.ReLU())
                ]))

    4、卷积：
        卷积核尺寸以及常见卷积核：https://zhuanlan.zhihu.com/p/25754846 
        卷积运算的三个重要思想：稀疏交互（sparse interaction）、参数共享（parameter sharing）等变表示（equivalent representations）
        卷积操作变体参看: https://github.com/vdumoulin/conv_arithmetic

    5、batchnorm：（num_features，eps = 1e-05，momentum = 0.1，affine = True，track_running_stats = True ）
        Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift 
        批量标准化：通过减少内部协变量偏移来加速深度网络训练
        https://zhuanlan.zhihu.com/p/24810318

    6、tensor改变形状
        y = x.reshape([batchsize, -1, sentsize, wordsize])，-1表示自动根据其他维度进行计算
        tensor的操作非常、非常丰富，详见https://pytorch.org/docs/stable/tensors.html
        较常用操作包括：
        b = a.view(-1, 3)
        b.unsqueeze(1) 在第1维（下标从0开始）上增加“１” 
        b.expand_as(a)
        b.reshape_as(a)
'''

model = ConvNet(num_classes).to(device)

criterion = nn.CrossEntropyLoss()
'''
    1、没有衡量就没有改进。损失函数即起衡量作用。
    2、torch.nn.functional.cross_entropy(input,     输入值，即训练值
                                        target,     目标值，即标签值
                                        weight=None,    
                                        size_average=None,
                                        ignore_index=-100,
                                        reduce=None,
                                        reduction='mean')

    3、loss function：常用均方误差、极大似然、交叉熵
        简介https://blog.algorithmia.com/introduction-to-loss-functions/
        pytorch中lossfunction种类繁多，参见https://pytorch.org/docs/stable/nn.html#loss-functions
    4、损失函数选取原则（专题准备中）
'''

optimizer = torch.optim.Adam(model.patameter(),lr = learning_rate)
'''
    1、优化器种类多，Adam大法好：
        torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        params（iterable） - 可迭代的参数，用于优化或决定参数组
        lr（float，optional） - 学习率（默认1e-3）
        betas（Tuple [ float，float ] ，optional） - 用于计算梯度及其平方的运行平均值的系数（默认（0.9,0.999））
        eps（float，optional） - 添加到分母中以增强数值稳定性的术语（默认1e-8）
        weight_decay（float，optional） - 权重衰减（L2惩罚）（默认0）
        amsgrad（boolean ，optional） （默认False）

    2、至此准备工作已完成，再往下都是训练。
'''

total_step = len(train_loader)

for epoch in range(total_step):
    for i,(images,labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs,labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1)%100 == 0:
            print('Epoch [{}/{}],Step [{}/{}].Loss:{:.4f}'
            .format(epoch+1,num_epochs,i+1,total_step,loss.item()))
'''
    1、所有optimizer优化器都包含一个step()函数，用于执行单个优化步骤，实现参数更新。
        调用方式有两种：
        optimizer.step()  ：简化版本，自动调用backward
        optimizer.step(closure)  ：允许重新计算模型的闭包，参见https://pytorch.org/docs/stable/optim.html#optimizer-step
'''

model.eval()
'''
    eval（）基类Module的成员函数，将模型设置为evaluation模式（测试模式），只对特定的模块类型有效，如Dropout和BatchNorm等
'''

with torch.no_grad():
    correct = 0
    total = 0
    for images,labels in test_loader:
        imgsges = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _,predicted = torch.max(outputs.data,1)
        total +=labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the 1000 test images: {} %'.format(100 * correct/total))

torch.save(model.state_dict(),'model.ckpt')

'''
    1、torch.autograd.no_grad,禁用自动梯度计算的上下文管理器
        torch.autograd.enable_grad，支持梯度计算
        torch.autograd.set_grad_enabled（mode ），mode可以为True或False
    2、torch.max（input，dim，keepdim = False，out = None）
        input——Tensor
        dim——int，要减少的维度（the dimension to reduce）
        keepdim（bool，optional） - 输出张量是否dim保留。默认False。
        out（元组，可选） - 两个输出张量的结果元组（max，max_indices）
    3、 _,predicted = torch.max(outputs.data,1)怎么解释？？？尚未理解



    4、模型存储和加载
        API:
        save(obj, f, pickle_module=pickle, pickle_protocol=DEFAULT_PROTOCOL):
        load(f, map_location=None, pickle_module=pickle):

        模型存储和加载的两种方法
        torch.save(the_model.state_dict(), PATH)
        the_model = TheModelClass(*args, **kwargs)
        the_model.load_state_dict(torch.load(PATH))

        torch.save(the_model, PATH)
        the_model = torch.load(PATH)

'''