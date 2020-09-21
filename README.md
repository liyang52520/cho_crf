# cho_crf

## 代码

1. master：普通unigram和bigram代码
2. combine：将unigram和bigram结合
3. minus_span：使用minus_span，即对于span(i, j)，用(f_j - f_i) \concat (b_i - b_j)表示
4. biaffine: 使用biaffine替代mlp进行打分

## res
结果见[wiki](http://120.132.13.131:8080/wiki/index.php/CHOCRF-liyang)

