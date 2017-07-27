require("mxnet")

source("mx.io.bucket.iter.R")
source("rnn_bucket_setup.R")
source("rnn_bucket_train.R")

corpus_bucketed_train <-
  readRDS(file = "corpus_bucketed_train_100_200_300_500_800_left.rds")
corpus_bucketed_test <-
  readRDS(file = "corpus_bucketed_test_100_200_300_500_800_left.rds")

vocab <- length(corpus_bucketed_test$dic)

### Create iterators
batch_size = 64

train.data <- mx_io_bucket_iter(
  buckets = corpus_bucketed_train$buckets,
  batch_size = batch_size,
  data_mask_element = 0,
  shuffle = TRUE
)

eval.data <- mx_io_bucket_iter(
  buckets = corpus_bucketed_test$buckets,
  batch_size = batch_size,
  data_mask_element = 0,
  shuffle = FALSE
)

num.label = 2
num.embed = 16
num.hidden = 24
update.period = 1

metric <- mx.metric.accuracy

input.size = vocab
initializer = mx.init.Xavier(rnd_type = "gaussian",
                             factor_type = "in",
                             magnitude = 2)
dropout = 0.25
verbose = TRUE

devices <- list(mx.cpu())
end.round = 16

optimizer <- mx.opt.create("adadelta",
                           rho = 0.92,
                           epsilon = 1e-6,
                           wd = 0.0002,
                           clip_gradient = NULL,
                           rescale.grad = 1 / batch_size)

num.rnn.layer <- 2

model_sentiment_lstm <- mx.rnn.buckets(train.data =  train.data,
                                       eval.data = eval.data,
                                       begin.round = 1,
                                       end.round = end.round,
                                       ctx = devices,
                                       metric = metric,
                                       optimizer = optimizer,
                                       kvstore = "local",
                                       num.rnn.layer = num.rnn.layer,
                                       num.embed = num.embed,
                                       num.hidden = num.hidden,
                                       num.label = num.label,
                                       input.size = input.size,
                                       update.period = 1,
                                       initializer = initializer,
                                       dropout = dropout,
                                       config = "seq-to-one",
                                       batch.end.callback = mx.callback.log.train.metric(period = 50),
                                       epoch.end.callback = NULL,
                                       verbose = TRUE)

mx.model.save(model_sentiment_lstm, prefix = "model_sentiment_lstm", iteration = end.round)
