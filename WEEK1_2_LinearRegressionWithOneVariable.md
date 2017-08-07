# WEEK1 : Linear Regression with One Variable

## Model Representation
Our first learning algorithm will be **linear regression**. In this video, you'll see what the model looks like and more importantly you'll see what the overall process of supervised learning looks like.

最初に学ぶ学習アルゴリズムは**線形回帰**です。このビデオでは、そのモデルがどのようなもので、さらに重要なのは、総合的に教師あり学習のプロセスがどのようなものかを見ていきます。


Let's use some motivating example of predicting housing prices. We're going to use a data set of housing prices from the city of Portland, Oregon. And here I'm gonna plot my data set of a number of houses that were different sizes that were sold for a range of different prices.

では、動機付けをする例として、住宅の価格を予測する例を使いましょう。ここで使うデータセットは、 オレゴン州ポートランド市の住宅価格のものです。ここにデータセットをプロットします。いくつかの家の それぞれのサイズと、それらが売れたそれぞれの価格の範囲です。


Let's say that given this data set, you have a friend that's trying to sell a house and let's see if friend's house is size of 1250 square feet and you want to tell them how much they might be able to sell the house for.

このデータセットを元に、家を売ろうとしている友人がいたとして、そうですね、その人の家のサイズが 1250平行フィートで、 幾らで家を売ることが出来るか教えたいとします。


Well one thing you could do is fit a model. Maybe fit a straight line to this data. Looks something like that and based on that, maybe you could tell your friend that let's say maybe he can sell the house for around $220,000. So this is an example of a supervised learning algorithm. And it's supervised learning because we're given the, quotes, "right answer" for each of our examples.

さて、できることの一つは、モデルを当てはめることです。 例えば、このデータに直線を当てはめるとします。こんな感じに。そして それに基づき、友人に言えることは、そうですね、例えばその家をだいたい 22万ドルぐらいで売れるだろうということです。これは 教師あり学習アルゴリズムの一例です。これが教師あり学習だという理由はそれぞれのサンプルに対して「正解」が与えられているからです。


Namely we're told what was the actual house, what was the actual price of each of the houses in our data set were sold for and moreover, this is an example of a regression problem where the term regression refers to the fact that we are predicting a real-valued output namely the price. And just to remind you the other most common type of supervised learning problem is called the classification problem where we predict discrete-valued outputs such as if we are looking at cancer tumors and trying to decide if a tumor is malignant or benign.

つまり、 実際の家が、データセット内のそれぞれの家が実際にいくらで売れたのかです。さらに、これは回帰問題の一例です。 回帰という言葉は、私たちが予測しようとしているのが実数値の出力であるという事実を指しています。 つまり価格です。そして思い返して頂きたいのは、もう一つの最も一般的なタイプの 教師あり学習問題は分類問題と呼ばれ、それは離散値出力を予測するものでした。例えば、癌腫瘍を見て どれが悪性でどれが良性かを判別する場合です。


So that's a zero-one valued discrete output. More formally, in supervised learning, we have a data set and this data set is called a training set. So for housing prices example, we have a training set of different housing prices and our job is to learn from this data how to predict prices of the houses.

それは 0 か 1 の値で、つまり離散値です。もっと形式的に言うと、教師あり学習ではデータセットがあり、そしてこのデータセットは訓練セットと呼ばれます。ですから住宅の価格の例では、訓練セットとして それぞれの家の価格があり、課題は、このデータから住宅の価格を予測するよう学習することです。


Let's define some notation that we're using throughout this course. We're going to define quite a lot of symbols. It's okay if you don't remember all the symbols right now but as the course progresses it will be useful [inaudible] convenient notation.

では、このコースを通して使ういくつかの表記方法を定義します。 かなりの数のシンボルを定義していきます。 全てのシンボルを 今すぐ覚えなくても構いません。しかし、コースが進行するにつれて、こうした表記方法を覚えていくと便利です。


So I'm gonna use lower case m throughout this course to denote the number of training examples. So in this data set, if I have, you know, let's say 47 rows in this table. Then I have 47 training examples and m equals 47. 

では、このコースを通して小文字の m を使って 訓練サンプルの数を表します。ですからの、このデータセットでは この表には 47 行あるとします。ということは 47 件の訓練サンプルがあるということで、m = 47 となります。


Let me use lowercase x to denote the input variables often also called the features. That would be the x is here, it would the input features. 

小文字の x を使って、入力変数を表します。これはよく 特徴とも言われます。これはここの x です。 この下に並ぶのが入力する特徴です。


And I'm gonna use y to denote my output variables or the target variable which I'm going to predict and so that's the second column here.

そして y を使って予測する出力変数、目標変数を表します。 そしてそれはこの二番目の列です。


[inaudible] notation, I'm going to use (x, y) to denote a single training example. So, a single row in this table corresponds to a single training example and to refer to a specific training example, I'm going to use this notation x(i) comma gives me y(i) And, we're going to use this to refer to the ith training example.

もう少し表記方法について。 (x, y) と表記して一件の訓練サンプルを表します。ですから、この表の各行が 一件の訓練サンプルに対応します。そして、特定の 訓練サンプルを指すときには、この表記方法 (x(i), y(i)) を使い、これを使って i 番目の訓練サンプルを指します。


So this superscript i over here, this is not exponentiation right? This (x(i), y(i)), the superscript i in parentheses that's just an index into my training set and refers to the ith row in this table, okay? So this is not x to the power of i, y to the power of i. Instead (x(i), y(i)) just refers to the ith row of this table.

ですから、この添え字 i は ここにありますが、これは指数ではありません。よいですか。この (x(i), y(i)) では、添え字の i は 括弧で囲まれており、それは単に訓練セットへのインデックスで、 このテーブルの i 行目を示します。いいですね。ですから、これは x の i 乗、y の i 乗という意味ではありません。 (x(i), y(i)) は単にこの表の i 番目の列を示すだけです。


So for example, x(1) refers to the input value for the first training example so that's 2104. That's this x in the first row. x(2) will be equal to 1416 right? That's the second x and y(1) will be equal to 460. The first, the y value for my first training example, that's what that (1) refers to. 

つまり、例えば、x(1) は 最初の訓練サンプルの入力値を指しますので、それは 2104 です。それは、この 最初の行の x の 値です。x(2) は = 1416、ですね。これが二番目の x の値です。そして y(1) は = 460、最初の y の値、 最初の訓練サンプルです。それが(1) の示す意味です。

So as mentioned, occasionally I'll ask you a question to let you check your understanding and a few seconds in this video a multiple-choice question will pop up in the video. When it does, please use your mouse to select what you think is the right answer. 

前にも触れた通り、時々、 質問をして、皆さんが自分の理解を確認できるようにします。数秒後にこの ビデオに選択問題がビデオに表示されます。 そうしたら、マウスを使って、あなたが正しいと思う答えを選択してください。 

(ここで問題が表示される)


What defined by the training set is. So here's how this supervised learning algorithm works.

訓練セットによって定義されるのは何か。これが教師あり学習アルゴリズムの仕組みです。


We saw that with the training set like our training set of housing prices and we feed that to our learning algorithm. Is the job of a learning algorithm to then output a function which by convention is usually denoted lowercase h and h stands for hypothesis And what the job of the hypothesis is, is, is a function that takes as input the size of a house like maybe the size of the new house your friend's trying to sell so it takes in the value of x and it tries to output the estimated value of y for the corresponding house. So h is a function that maps from x's to y's.

開始点に住宅価格の訓練セットような訓練セットがあります。そしてそれを学習アルゴリズムに読み込ませます。 学習アルゴリズムの仕事は、ある関数を出力することで、慣習的にこれは小文字の h で表記されます。 h は 仮説（hypothesis）の略です。そして仮説の仕事は、関数として入力として家のサイズを受け取り、例えば、友人が売ろうとしている新しい家のサイズなど、そして与えられた x の値に対してそれに相当する家の推定価格 y を出力しようとします。つまり、h は x から y に対応付けする関数です。


People often ask me, you know, why is this function called hypothesis. Some of you may know the meaning of the term hypothesis, from the dictionary or from science or whatever. It turns out that in machine learning, this is a name that was used in the early days of machine learning and it kinda stuck.

よく人から聞かれるのは、なぜこの関数が 仮説と呼ばれるかです。皆さんの中には仮説という言葉の意味をご存知の方もいると思います。 辞書からとか、あるいは科学からとか何とか。実は、機械学習では、これは 機械学習の初期に使われた名前で、それがなんとなく定着してしまったのです。


'Cause maybe not a great name for this sort of function, for mapping from sizes of houses to the predictions, that you know.... I think the term hypothesis, maybe isn't the best possible name for this, but this is the standard terminology that people use in machine learning. So don't worry too much about why people call it that. 

このような関数にはそれほどふさわしい名前ではないかもしれません。家のサイズを 予測価格に対応付けするような場合には。仮説という用語は多分 一番適切な名前ではないでしょう。しかし、人々が使う標準的な用語として、 機械学習では使われています。ですから、なぜ人々がそのように呼ぶのかは、あまり気にしないで下さい。


When designing a learning algorithm, the next thing we need to decide is how do we represent this hypothesis h. For this and the next few videos, I'm going to choose our initial choice , for representing the hypothesis, will be the following. We're going to represent h as follows. And we will write this as h<u>theta(x) equals theta<u>0</u></u> plus theta<u>1 of x. And as a shorthand, sometimes instead of writing, you</u> know, h subscript theta of x, sometimes there's a shorthand, I'll just write as a h of x. But more often I'll write it as a subscript theta over there.

学習アルゴリズムを設計する際に、次に決めなければいけないことは、どのようにこの仮説 h を表現するかです。 今回以降の数回のビデオでは 初期の選択として、仮説の表現を以下のようにします。 h を以下のように表現します。 そしてこれをこのように書きます。hθ(x) = θ0 + θ1 掛ける x。そして簡略表記として、 hθ(x) と書く代わりに、h(x) と簡略して書くことがあります。 しかし、たいていの場合は、添え字のthetaをそこに書きます。


And plotting this in the pictures, all this means is that, we are going to predict that y is a linear function of x. Right, so that's the data set and what this function is doing, is predicting that y is some straight line function of x. That's h of x equals theta 0 plus theta 1 x, okay? And why a linear function? Well, sometimes we'll want to fit more complicated, perhaps non-linear functions as well. But since this linear case is the simple building block, we will start with this example first of fitting linear functions, and we will build on this to eventually have more complex models, and more complex learning algorithms. Let me also give this particular model a name. This model is called linear regression or this, for example, is actually linear regression with one variable, with the variable being x. Predicting all the prices as functions of one variable X. And another name for this model is univariate linear regression. And univariate is just a fancy way of saying one variable. So, that's linear regression. In the next video we'll start to talk about just how we go about implementing this model. 

そしてこれを図としてプロットすると、これの意味は y を x の線形関数として予測するということです。これがデータセットで、この関数が行っているのは、y がなんらかの x の直線の関数だと予測しているわけです。h(x) = θ0 + θ1 * x。 分かりますか。ではなぜ線形関数なのか。時には、もっと複雑な、例えば非線形関数を当てはめたいこともあります。しかし、この線形のケースは 簡単な基本となるケースですので、まずはこの例では最初に線形関数に当てはめ、いずれはこれを発展させてもっと複雑な モデル、もっと複雑な学習アルゴリズムにしていきます。また、 このモデルを特に指す名前をつけたいと思います。このモデルは**線形回帰**といい、 この例は、実は変数が一つの場合の線形回帰で、その変数は x です。 全ての価格を一つの変数 x で予測するものです。そして このモデルのもう一つの名前は**単回帰**です。そして単回帰の単は 変数が一つであるということを意味しています。これが線形回帰です。次の ビデオでは、このモデルをどのように実装していくかをお話します。 


## Cost Function

## Cost Function - Intuition I

## Cost Function - Intuition II
