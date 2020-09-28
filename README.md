# cho_crf

## 代码

1. master：普通unigram和bigram代码
2. combine(improve 1)：将unigram和bigram结合
3. minus_span(improve 2)：使用minus_span，即对于span(i, j)，用(f_j - f_i) \concat (b_i - b_j)表示
4. improve_3: 在使用bigram时，分别计算了ℎ_(𝑖−1)和ℎ_𝑖到n_labels个标签的发射分值（使用了不同的BiLSTM和MLP，想计算不同的ℎ_𝑖 处于不同的作用时的发射分值），bigram的分值仍然当作h_(𝑖−1) 转移到h_𝑖的分值，将三者结合计算；
5. improve_4: 使用biaffine替换MLP层进行打分，计算其到n_labels个标签的分值

## res
结果见[wiki](http://120.132.13.131:8080/wiki/index.php/CHOCRF-liyang)

