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


source("rnn.infer.R")

infer_iter = eval.data

model = mx.model.load("model_sentiment_lstm", iteration = 3)

ctx = mx.cpu()

output_last_state = FALSE

init.state = NULL

cell.type = "lstm"

config = "seq-to-one"

pred <- mx.rnn.infer.buckets(infer_iter = eval.data,
                             model,
                             config,
                             ctx = mx.cpu()) 
