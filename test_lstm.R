require("mxnet")

source("mx.io.bucket.iter.R")
source("rnn.R")

corpus_bucketed_train <-
  readRDS(file = "corpus_bucketed_train_100_200_300_500_800_left.rds")
corpus_bucketed_test <-
  readRDS(file = "corpus_bucketed_test_100_200_300_500_800_left.rds")

vocab <- length(corpus_bucketed_test$dic)

### Create iterators
batch_size = 64

train.data <-
  mx.io.bucket.iter(
    buckets = corpus_bucketed_train$buckets,
    batch.size = batch_size,
    data.mask.element = 0,
    shuffle = TRUE
  )

eval.data <-
  mx.io.bucket.iter(
    buckets = corpus_bucketed_test$buckets,
    batch.size = batch_size,
    data.mask.element = 0,
    shuffle = FALSE
  )

num.label = 2
num.embed = 16
num.hidden = 24

metric <- mx.metric.accuracy

input.size = vocab
initializer = mx.init.Xavier(rnd_type = "gaussian",
                             factor_type = "in",
                             magnitude = 2)
dropout = 0.25
verbose = TRUE

end.round = 1

optimizer <- mx.opt.create(
  "adadelta",
  rho = 0.92,
  epsilon = 1e-6,
  wd = 0.0002,
  clip_gradient = NULL,
  rescale.grad = 1 / batch_size
)

num.rnn.layer <- 2

cell.type = "lstm"

config = "seq-to-one"

ctx = mx.cpu()

kvstore = "local"

mx.set.seed(42)

model_sentiment_lstm <- mx.rnn.buckets(
  train.data =  train.data,
#  eval.data = eval.data,
  begin.round = 1,
  end.round = end.round,
  ctx = mx.cpu(),
  metric = metric,
  optimizer = optimizer,
  kvstore = "local",
  num.rnn.layer = num.rnn.layer,
  num.embed = num.embed,
  num.hidden = num.hidden,
  num.label = num.label,
  input.size = input.size,
  initializer = initializer,
  dropout = dropout,
  config = "seq-to-one",
  batch.end.callback = mx.callback.log.train.metric(period = 50),
  verbose = TRUE
)

mx.model.save(model_sentiment_lstm,
              prefix = "model_sentiment_lstm",
              iteration = end.round)
