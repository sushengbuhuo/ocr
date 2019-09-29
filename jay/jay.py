#pip install wordcloud ,jieba
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import jieba,wordcloud
from snownlp import SnowNLP
import jieba.analyse
t="""
没有了联络 后来的生活
我都是听别人说
说你怎么了 说你怎么过
放不下的人是我
人多的时候 就待在角落
就怕别人问起我
你们怎么了 你低着头
护着我连抱怨都没有
电话开始躲 从不对我说
不习惯一个人生活
离开我以后 要我好好过
怕打扰想自由的我
都这个时候 你还在意着
别人是怎么怎么看我的
拼命解释着
不是我的错 是你要走
眼看着你难过
挽留的话却没有说
你会微笑放手
说好不哭让我走
电话开始躲 从不对我说
不习惯一个人生活
离开我以后 要我好好过
怕打扰想自由的我
都这个时候 你还在意着
别人是怎么怎么看我的
拼命解释着
不是我的错 是你要走
眼看着你难过
挽留的话却没有说
你会微笑放手
说好不哭让我走
你什么都没有
却还为我的梦加油
心疼过了多久
过了多久
还在找理由等我
"""
#选取周杰伦的《晴天》歌词
mytext = """
故事的小黄花
从出生那年就飘着
童年的荡秋千
随记忆一直晃到现在
ㄖㄨㄟ ㄙㄡ ㄙㄡ ㄒ一 ㄉㄡ ㄒ一ㄌㄚ
Re So So Si Do Si La
ㄙㄡ ㄌㄚ ㄒ一 ㄒ一 ㄒ一 ㄒ一 ㄌㄚ ㄒ一 ㄌㄚ ㄙㄡ
So La Si Si Si Si La Si La So
吹着前奏望着天空

我想起花瓣试着掉落

为你翘课的那一天

花落的那一天

教室的那一间

我怎么看不见

消失的下雨天

我好想再淋一遍

没想到失去的勇气我还留着

好想再问一遍

你会等待还是离开

刮风这天我试过握着你手

但偏偏雨渐渐大到我看你不见

还要多久我才能在你身边

等到放晴的那天也许我会比较好一点

从前从前有个人爱你很久

但偏偏风渐渐把距离吹得好远

好不容易又能再多爱一天

但故事的最后你好像还是说了拜拜



为你翘课的那一天

花落的那一天

教室的那一间

我怎么看不见

消失的下雨天

我好想再淋一遍

没想到失去的勇气我还留着

好想再问一遍

你会等待还是离开

刮风这天我试过握着你手

但偏偏雨渐渐大到我看你不见

还要多久我才能在你身边

等到放晴的那天也许我会比较好一点

从前从前有个人爱你很久

偏偏风渐渐把距离吹得好远

好不容易又能再多爱一天

但故事的最后你好像还是说了拜拜

刮风这天我试过握着你手

但偏偏雨渐渐大到我看你不见

还要多久我才能够在你身边

等到放晴那天也许我会比较好一点

从前从前有个人爱你很久

但偏偏雨渐渐把距离吹得好远

好不容易又能再多爱一天

但故事的最后你好像还是说了拜
"""
mytext = " ".join(jieba.cut(t))
from collections import Counter
c = Counter(mytext)
c = c.most_common(10)

#[(' ', 522), ('\n', 98), ('一', 24), ('我', 19), ('你', 19), ('的', 17), ('天', 16), ('好', 16), ('偏',12), ('渐', 12)]

# wordcloud = WordCloud(font_path="c:\windows\fonts\simhei.ttf").generate(mytext)
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis("off")
# plt.show()
# 保存图片
#wordcloud.to_file('test.jpg')

#每个句子分析
s = SnowNLP(mytext)
# for sentence in s2.sentences:
# 	print(sentence)
# 	sentc = SnowNLP(sentence)
# 	print(sentc.sentiments)

#wordcloud_cli --text no_cry.txt --imagefile no_cry.png --mask no_cry.jpeg  --fontfile c:\windos\fonts\simhei.ttf
#第一句的情感分析结果
s1 = SnowNLP(s.sentences[0])
#s1.sentiments
#0.8849970682062196#正向情感
#分析 好不容易 又 能 再 多 爱 一天
s1 = SnowNLP(s.sentences[-2])
#s1.sentiments
#0.21646625648493734#这个情绪就比较负面了

def handle(textfile, stopword):
    with open(textfile, 'r',encoding='utf-8') as f:
       data = f.read()
    wordlist = jieba.analyse.extract_tags(data, topK=50)   # 分词，取前100
    wordStr = " ".join(wordlist)
    # print (wordStr)
    #怎么 别人 挽留 打扰 放手 说好 在意 没有 从不 拼命 难过 多久 眼看 生活 微笑 好好 时候 习惯 自由 电话 解释 离开 的话 护着 放不下 加油 以后 问起 人多 角落 心疼 低着头 抱怨 开始 不是 联络 这个 理由 一个 后来 你们 什么
    hand = np.array(Image.open('ye.jpg'))    # 打开一张图片，词语以图片形状为背景分布
    my_cloudword = WordCloud(
        # wordcloud参数配置
        width=500,
        height=500,
        background_color = 'white',   # 背景颜色设置白色
        #mask = hand,                  # 背景图片
        max_words = 100,              # 最大显示的字数
        stopwords = stopword,         # 停用词
        max_font_size = 100,           # 字体最大值
        font_path='c:\windows\fonts\simhei.ttf',  # 设置中文字体，若是有中文的话，这句代码必须添加，不然会出现方框，不出现汉字
        random_state=3,  # 设置有多少种随机生成状态，即有多少种配色方案
    )
    my_cloudword.generate(wordStr)          # 生成图片
    #my_cloudword.to_file('res.jpg')    # 保存
    plt.axis('off')  # 是否显示x轴、y轴下标
    ax = plt.imshow(my_cloudword)  # 显示词云图
    fig = ax.figure
    fig.set_size_inches(25,20)                  # 可调节图片紧密 尺寸程度    
    plt.show()  # 显示
    # df = pd.DataFrame([x.split(';') for x in data.split('\r\n')])
    # semiscore = data.apply(lambda x: SnowNLP(x).sentiments)
    # semilabel = semiscore.apply(lambda x: 1 if x>0.5 else -1)
    # plt.hist(semiscore, bins = np.arange(0, 1.01, 0.01),label="semisocre", color="#ff9999")
    # plt.xlabel("semiscore")
    # plt.ylabel("number")
    # plt.title("The semi-score of comment")
    # plt.show()

    # semilabel = semilabel.value_counts()
    # plt.bar(semilabel.index,semilabel.values,tick_label=semilabel.index,color='#90EE90')
    # plt.xlabel("semislabel")
    # plt.ylabel("number")
    # plt.title("The semi-label of comment")
    # plt.show()
    sentimentslist = []
    s = SnowNLP(data)
    for sentence in s.sentences:
        s = SnowNLP(sentence)
        sentimentslist.append(s.sentiments)
    plt.hist(sentimentslist, bins = np.arange(0, 1, 0.01), facecolor = 'g')
    plt.xlabel('Sentiments Probability')
    plt.ylabel('Quantity')
    plt.title('Analysis of Sentiments')
    plt.show()


stopwords = set(STOPWORDS)
handle('no_cry.txt', stopwords)
