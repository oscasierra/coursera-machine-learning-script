# WEEK3 : Logistic Regression
---
## Classfication and Representation
### Classification (分類問題)
In this and the next few videos, I want to start to talk about classification problems, where the variable y that you want to predict is valued.

このビデオと続くいくつかのビデオでは、**分類問題**について話していきます。 分類とは予測したい値 y が離散的な値の時の話です。


We'll develop an algorithm called logistic regression, which is one of the most popular and most widely used learning algorithms today. 

**ロジスティック回帰**と呼ばれるアルゴリズムを開発していきます。 それはこんにち、機械学習の分野では、もっともポピュラーでもっとも広く使われているアルゴリズムの一つです。


Here are some examples of classification problems. Earlier we talked about email spam classification as an example of a classification problem.

これが分類問題の例です。以前に、メールのスパム分類を分類問題の例だと言ったと思います。 


Another example would be classifying online transactions. So if you have a website that sells stuff and if you want to know if a particular transaction is fraudulent or not, whether someone is using a stolen credit card or has stolen the user's password. There's another classification problem.

他の例としては、オンラインの売買を分類する、などが考えられる。 たとえばあなたが物を売るWEBサイトを持っているとして、実際の売買が詐欺かどうかを知りたいとする。例えば誰かが盗まれたクレジットカードを使っているかどうか、盗まれたパスワードを使っているかどうか。それも分類問題です。


And earlier we also talked about the example of classifying tumors as cancerous, malignant or as benign tumors. In all of these problems the variable that we're trying to predict is a variable y that we can think of as taking on two values either zero or one, either spam or not spam, fraudulent or not fraudulent, related malignant or benign. Another name for the class that we denote with zero is the negative class, and another name for the class that we denote with one is the positive class. So zero we denote as the benign tumor, and one, positive class we denote a malignant tumor. The assignment of the two classes, spam not spam and so on. The assignment of the two classes to positive and negative to zero and one is somewhat arbitrary and it doesn't really matter but often there is this intuition that a negative class is conveying the absence of something like the absence of a malignant tumor. Whereas one the positive class is conveying the presence of something that we may be looking for, but the definition of which is negative and which is positive is somewhat arbitrary and it doesn't matter that much. 

そして以前、腫瘍をガンになりうる悪性の物か良性の腫瘍かの分類の例も話しました。これら全ての問題で、我らが予想したい変数は変数Yで、それは2つの値のどちらかをとると考えられる、 0か1か、とか、スパムかスパムじゃないか、とか、詐称か詐称でないかとか、悪性か良性かとか。 0 で表しているクラス(分類)のもう一つの名前は陰性(ネガティブクラス)で、1 で表しているクラスのもう一つの呼び名は陽性(ポジティブクラス)だ。つまり 0 は良性の腫瘍を表し、1つまり陽性は、悪性の腫瘍を表すなど。 2つのクラスを割り振る、スパムかスパムじゃないかなどなど。2つのクラス、陽性か陰性か 0 か 1 か、に何を割り振るかはなんでも良くて、任意。そこはどうでも良い。でも良く、陰性は何かが無い、例えば悪性の腫瘍が無い、などの不在っぽい感覚を伝える。 他方、陽性は何かが存在してる感じを語感として持つ、我らの探している何かの存在。 だが何が陰性で何が陽性かの定義は任意で、それはどうでも良い。

For now we're going to start with classification problems with just two classes zero and one. Later one we'll talk about multi class problems as well where therefore y may take on four values zero, one, two, and three. This is called a multiclass classification problem. But for the next few videos, let's start with the two class or the binary classification problem and we'll worry about the multiclass setting later. So how do we develop a classification algorithm? 

さて、まずは2つのクラス「0」と「1」だけがあるケースの分類問題から始めよう。多数のクラスの問題、たとえば Y が 0、1、2、3 の値をとるようなケースについては、その後に扱おう。これは**マルチクラス**の分類問題と呼ばれる。 だがここからの2、3のビデオでは2つのクラスだけ、つまり**バイナリ分類問題**から始めよう。そしてマルチクラスの話はその後に考えることとする。では、どのように分類アルゴリズムを作るか？


Here's an example of a training set for a classification task for classifying a tumor as malignant or benign. And notice that malignancy takes on only two values, zero or no, one or yes. 

ここに、腫瘍を悪性か良性か分類する分類タスクのトレーニングセットがある。見ての通り、malignancy(悪性)の値は2つの値だけをとる、つまり 0 (No)か、1 (Yes)かだ。


So one thing we could do given this training set is to apply the algorithm that we already know. Linear regression to this data set and just try to fit the straight line to the data. 

これらのトレーニングセットが与えられた時に我々が出来る事の一つは、既に知ってるアルゴリズムを適用することだ。 線形回帰をこのデータセットに適用して、単に直線をこのデータにフィッティングさせる。


So if you take this training set and fill a straight line to it, maybe you get a hypothesis that looks like that, right. So that's my hypothesis.

もし、このトレーニングセットをとり、そこに直線をフィットさせると、たぶんえられる仮説はこのようになる。 これは私の仮説だ。


 H(x) equals theta transpose x. If you want to make predictions one thing you could try doing is then threshold the classifier outputs at 0.5 that is at a vertical axis value 0.5 and if the hypothesis outputs a value that is greater than equal to 0.5 you can take y = 1. If it's less than 0.5 you can take y=0. Let's see what happens if we do that. 

h(x) = θ transpose x に一致する。もしも予測をしたいなら、ひとつ試してみるべきことは分類器の閾値を 0.5 に設定することだ。 これが0.5の値の縦線だ。そしてもし仮説が 0.5 以上の値を出力すれば yは 1 だと予測される。もし 0.5 未満であれば、yは 0 だと予測される。 こうやったときに何がおこるか見てみよう。


So 0.5 and so that's where the threshold is and that's using linear regression this way. Everything to the right of this point we will end up predicting as the positive cross. Because the output values is greater than 0.5 on the vertical axis and everything to the left of that point we will end up predicting as a negative value. 

さて、0.5を持ってきてみようそれはまさに閾値のある所だ。 つまるところ、このように線形回帰を使うと、 この点より右にある全ての点は結局ポジティブなクラスと予測する事となる、何故ならアウトプットの値は縦軸上では 0.5より大きいから。そしてその点より左の全ての点は結局の所、ネガティブ、と予測する事となる。 


In this particular example, it looks like linear regression is actually doing something reasonable. Even though this is a classification toss we're interested in. But now let's try changing the problem a bit. Let me extend out the horizontal access a little bit and let's say we got one more training example way out there on the right. Notice that that additional training example, this one out here, it doesn't actually change anything, right. 

この特定の例の場合、線形回帰は実際にリーズナブルと言えない事も無い事をしている、興味を持ってる事が分類問題であるとはいえども。 だが、ここでちょっとだけ問題を変更してみよう。 横軸をちょっと延長して もう一つ追加のトレーニングセットの例が右側にあるとしよう。 追加のトレーニングの例は見ての通り、ここある。 それは実際には何も違いは無い。でしょ？

Looking at the training set it's pretty clear what a good hypothesis is. Is that well everything to the right of somewhere around here, to the right of this we should predict this positive. Everything to the left we should probably predict as negative because from this training set, it looks like all the tumors larger than a certain value around here are malignant, and all the tumors smaller than that are not malignant, at least for this training set. 

But once we've added that extra example over here, if you now run linear regression, you instead get a straight line fit to the data. That might maybe look like this. 

And if you know threshold hypothesis at 0.5, you end up with a threshold that's around here, so that everything to the right of this point you predict as positive and everything to the left of that point you predict as negative. 

And this seems a pretty bad thing for linear regression to have done, right, because you know these are our positive examples, these are our negative examples. It's pretty clear we really should be separating the two somewhere around there, but somehow by adding one example way out here to the right, this example really isn't giving us any new information. I mean, there should be no surprise to the learning algorithm. That the example way out here turns out to be malignant. But somehow having that example out there caused linear regression to change its straight-line fit to the data from this magenta line out here to this blue line over here, and caused it to give us a worse hypothesis. 

So, applying linear regression to a classification problem often isn't a great idea. In the first example, before I added this extra training example, previously linear regression was just getting lucky and it got us a hypothesis that worked well for that particular example, but usually applying linear regression to a data set, you might get lucky but often it isn't a good idea. So I wouldn't use linear regression for classification problems. 

Here's one other funny thing about what would happen if we were to use linear regression for a classification problem. For classification we know that y is either zero or one. But if you are using linear regression where the hypothesis can output values that are much larger than one or less than zero, even if all of your training examples have labels y equals zero or one. 

And it seems kind of strange that even though we know that the labels should be zero, one it seems kind of strange if the algorithm can output values much larger than one or much smaller than zero. 

So what we'll do in the next few videos is develop an algorithm called logistic regression, which has the property that the output, the predictions of logistic regression are always between zero and one, and doesn't become bigger than one or become less than zero. 

And by the way, logistic regression is, and we will use it as a classification algorithm, is some, maybe sometimes confusing that the term regression appears in this name even though logistic regression is actually a classification algorithm. But that's just a name it was given for historical reasons. So don't be confused by that logistic regression is actually a classification algorithm that we apply to settings where the label y is discrete value, when it's either zero or one. So hopefully you now know why, if you have a classification problem, using linear regression isn't a good idea. In the next video, we'll start working out the details of the logistic regression algorithm. 
