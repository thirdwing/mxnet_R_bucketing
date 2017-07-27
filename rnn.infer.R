
mx.rnn.infer.one.to_one <- function(infer_iter,
                                    model,
                                    pred_length,
                                    config,
                                    ctx = list(mx.cpu()),
                                    kvstore = NULL,
                                    output_last_state = FALSE,
                                    init.state) {
  ### Infer parameters from model
  num.rnn.layer = ((length(model$arg.params) - 3) / 4)
  num.hidden = dim(model$arg.params$l1.h2h.weight)[1]
  input.size = dim(model$arg.params$embed.weight)[2]
  num.embed = dim(model$arg.params$embed.weight)[1]
  num.label = dim(model$arg.params$cls.bias)
  
  ### Initialise the iterator
  infer_iter$init()
  infer_iter$reset()
  batch_size <- infer_iter$batch_size
  
  # get unrolled lstm symbol
  sym_list <- sapply(infer_iter$bucket_names, function(x) {
    rnn.unroll(
      num.rnn.layer = num.rnn.layer,
      num.hidden = num.hidden,
      seq.len = as.integer(x),
      input.size = input.size,
      num.embed = num.embed,
      num.label = num.label,
      config = config,
      dropout = 0,
      init.state = init.state,
      output_last_state = output_last_state
    )
  },
  simplify = F, USE.NAMES = T)
  
  symbol <- sym_list[[names(infer_iter$bucketID())]]
  
  arg.names <- symbol$arguments
  input.shape <- lapply(infer_iter$value(), dim)
  input.shape <- input.shape[names(input.shape) %in% arg.names]
  
  infer_shapes <- symbol$infer.shape(input.shape)
  arg.params <- model$arg.params
  aux.params <- model$aux.params
  
  #####################################################################
  ### The above preperation is essentially the same as for training
  ### Should consider modulising it
  #####################################################################
  
  #####################################################################
  ### Binding seq to executor and iteratively predict
  #####################################################################
  
  ndevice <- length(ctx)
  
  symbol <- sym_list[[names(infer_iter$bucketID())]]
  input.names <- names(input.shape)
  arg.names <- names(arg.params)
  
  # Grad request
  grad_req <- rep("null", length(symbol$arguments))
  grad_null_idx <- match(input.names, symbol$arguments)
  grad_req[grad_null_idx] <- "null"
  
  # Arg array order
  update_names <- c(input.names, arg.names)
  arg_update_idx <- match(symbol$arguments, update_names)
  
  # Initial input shapes - need to be adapted for multi-devices - divide highest dimension by device nb
  s <- sapply(input.shape, function(shape) {
    mx.nd.zeros(shape = shape, ctx = mx.cpu())
  })
  
  #####################################################
  ### Initial binding
  train.execs <- lapply(1:ndevice, function(i) {
    mxnet:::mx.symbol.bind(
      symbol = symbol,
      arg.arrays = c(s, arg.params)[arg_update_idx],
      aux.arrays = aux.params,
      ctx = ctx[[i]],
      grad.req = grad_req
    )
  })
  
  ### initialize the predict
  pred <- NULL
  label <- NULL
  
  for (i in 1:pred_length) {
    seq_len <- as.integer(names(infer_iter$bucketID()))
    
    # Get input data slice
    dlist <- infer_iter$value()
    
    # Slice inputs for multi-devices
    slices <- lapply(dlist[input.names], function(input) {
      mx.nd.SliceChannel(
        data = input,
        num_outputs = ndevice,
        axis = 0,
        squeeze_axis = F
      )
    })
    
    ### get the new symbol
    ### Bind the arguments and symbol for the BucketID
    symbol <- sym_list[[names(infer_iter$bucketID())]]
    
    train.execs <- lapply(1:ndevice, function(i) {
      if (ndevice > 1)
        s <- lapply(slices, function(x)
          x[[i]])
      else
        s <- slices
      mxnet:::mx.symbol.bind(
        symbol = symbol,
        arg.arrays = c(s, train.execs[[i]]$arg.arrays[arg.names])[arg_update_idx],
        aux.arrays = train.execs[[i]]$aux.arrays,
        ctx = ctx[[i]],
        grad.req = grad_req
      )
    })
    
    for (texec in train.execs) {
      mx.exec.forward(texec, is.train = FALSE)
    }
    
    if (config == "one-to-one") {
      # copy outputs to CPU
      out.preds <- lapply(train.execs, function(texec) {
        lapply(texec$ref.outputs, function(output) {
          mx.nd.copyto(output, mx.cpu())
        })
      })
      
      ### Only works for 1 device
      pred <- lapply(1:length(out.preds[[1]]), function(i) {
        rbind(pred[[i]], as.array(out.preds[[1]][[i]]))
      })
    } else if (config == "seq-to-one") {
      # copy outputs to CPU
      out.preds <- lapply(train.execs, function(texec) {
        mx.nd.copyto(texec$ref.outputs[[length(symbol$outputs)]], mx.cpu())
      })
      
      pred <- rbind(pred, matrix(sapply(1:ndevice, function(i) {
        t(as.matrix(out.preds[[i]]))
      }), nrow = batch_size))
      
      label <- c(label, sapply(1:ndevice, function(i) {
        if (ndevice == 1)
          as.numeric(as.array(mx.nd.Reshape(slices$label, shape = -1)))
        else
          as.numeric(as.array(mx.nd.Reshape(slices[[i]]$label, shape = -1)))
      }))
      
    }
  }
  
  if (config == "one-to-one") {
    return(pred)
  } else if (config == "one-to-one") {
    return(list(pred = pred, label = label))
  }
}



#' Training LSTM Unrolled Model with bucketing
#'
#' @param input_seq integer
#'      The initializing sequence
#' @param input_length integer
#'      The number of initializing elements
#' @param infer_length integer
#'      The number of infered elements
#' @param model The model from which to perform inference.
#' @param random Logical, Whether to infer based on modeled probabilities (T) or by selecting most likely outcome
#' @param ctx mx.context, optional
#'      The device used to perform training.
#' @param kvstore string, not currently supported
#'      The optimization method.
#' @return an integer vector corresponding to the encoded dictionnary
#'
#' @export
mx.rnn.infer.buckets <- function(infer_iter,
                                 model,
                                 config,
                                 ctx = list(mx.cpu()),
                                 kvstore = NULL,
                                 output_last_state = FALSE,
                                 init.state = NULL,
                                 cell.type = "lstm") {
  ### Infer parameters from model
  if (cell.type == "lstm") {
    num.rnn.layer = ((length(model$arg.params) - 3) / 4)
    num.hidden = dim(model$arg.params$l1.h2h.weight)[1]
  } else if (cell.type == "gru") {
    num.rnn.layer = ((length(model$arg.params) - 3) / 8)
    num.hidden = dim(model$arg.params$l1.gates.h2h.weight)[1]
  }
  
  input.size = dim(model$arg.params$embed.weight)[2]
  num.embed = dim(model$arg.params$embed.weight)[1]
  num.label = dim(model$arg.params$cls.bias)
  
  ### Initialise the iterator
  infer_iter$reset()
  batch_size <- infer_iter$batch_size
  
  # get unrolled lstm symbol
  sym_list <- sapply(infer_iter$bucket_names, function(x) {
    rnn.unroll(
      num.rnn.layer = num.rnn.layer,
      num.hidden = num.hidden,
      seq.len = as.integer(x),
      input.size = input.size,
      num.embed = num.embed,
      num.label = num.label,
      config = config,
      dropout = 0,
      init.state = init.state,
      cell.type = cell.type,
      output_last_state = output_last_state
    )
  },
  simplify = F, USE.NAMES = T)
  
  symbol <- sym_list[[names(infer_iter$bucketID())]]
  
  arg.names <- symbol$arguments
  input.shape <- lapply(infer_iter$value(), dim)
  input.shape <- input.shape[names(input.shape) %in% arg.names]
  
  infer_shapes <- symbol$infer.shape(input.shape)
  arg.params <- model$arg.params
  aux.params <- model$aux.params
  
  #####################################################################
  ### The above preperation is essentially the same as for training
  ### Should consider modulising it
  #####################################################################
  
  #####################################################################
  ### Binding seq to executor and iteratively predict
  #####################################################################
  
  ndevice <- length(ctx)
  
  symbol <- sym_list[[names(infer_iter$bucketID())]]
  input.names <- names(input.shape)
  arg.names <- names(arg.params)
  
  # Grad request
  grad_req <- rep("null", length(symbol$arguments))
  grad_null_idx <- match(input.names, symbol$arguments)
  grad_req[grad_null_idx] <- "null"
  
  # Arg array order
  update_names <- c(input.names, arg.names)
  arg_update_idx <- match(symbol$arguments, update_names)
  
  # Initial input shapes - need to be adapted for multi-devices - divide highest dimension by device nb
  s <- sapply(input.shape, function(shape) {
    mx.nd.zeros(shape = shape, ctx = mx.cpu())
  })
  
  #####################################################
  ### Initial binding
  train.execs <- lapply(1:ndevice, function(i) {
    mxnet:::mx.symbol.bind(
      symbol = symbol,
      arg.arrays = c(s, arg.params)[arg_update_idx],
      aux.arrays = aux.params,
      ctx = ctx[[i]],
      grad.req = grad_req
    )
  })
  
  ### initialize the predict
  pred <- NULL
  label <- NULL
  
  while (infer_iter$iter.next()) {
    seq_len <- as.integer(names(infer_iter$bucketID()))
    
    # Get input data slice
    dlist <- infer_iter$value()
    
    # Slice inputs for multi-devices
    slices <- lapply(dlist[input.names], function(input) {
      mx.nd.SliceChannel(
        data = input,
        num_outputs = ndevice,
        axis = 0,
        squeeze_axis = F
      )
    })
    
    ### get the new symbol
    ### Bind the arguments and symbol for the BucketID
    symbol <- sym_list[[names(infer_iter$bucketID())]]
    
    train.execs <- lapply(1:ndevice, function(i) {
      if (ndevice > 1)
        s <- lapply(slices, function(x)
          x[[i]])
      else
        s <- slices
      mxnet:::mx.symbol.bind(
        symbol = symbol,
        arg.arrays = c(s, train.execs[[i]]$arg.arrays[arg.names])[arg_update_idx],
        aux.arrays = train.execs[[i]]$aux.arrays,
        ctx = ctx[[i]],
        grad.req = grad_req
      )
    })
    
    for (texec in train.execs) {
      mx.exec.forward(texec, is.train = FALSE)
    }
    
    if (config == "one-to-one") {
      # copy outputs to CPU
      out.preds <- lapply(train.execs, function(texec) {
        lapply(texec$ref.outputs, function(output) {
          mx.nd.copyto(output, mx.cpu())
        })
      })
      
      ### Only works for 1 device
      pred <- lapply(1:length(out.preds[[1]]), function(i) {
        rbind(pred[[i]], as.array(out.preds[[1]][[i]]))
      })
    } else if (config == "seq-to-one") {
      # copy outputs to CPU
      out.preds <- lapply(train.execs, function(texec) {
        mx.nd.copyto(texec$ref.outputs[[length(symbol$outputs)]], mx.cpu())
      })
      
      pred <- rbind(pred, matrix(sapply(1:ndevice, function(i) {
        t(as.matrix(out.preds[[i]]))
      }), nrow = batch_size))
      
      label <- c(label, sapply(1:ndevice, function(i) {
        if (ndevice == 1)
          as.numeric(as.array(mx.nd.Reshape(slices$label, shape = -1)))
        else
          as.numeric(as.array(mx.nd.Reshape(slices[[i]]$label, shape = -1)))
      }))
      
    }
  }
  
  if (config == "one-to-one") {
    return(pred)
  } else if (config == "seq-to-one") {
    return(list(pred = pred, label = label))
  }
}



# Extract model from executors
mx.model.extract.model <- function(symbol, train.execs) {
  reduce.sum <- function(x)
    Reduce("+", x)
  # Get the parameters
  ndevice <- length(train.execs)
  narg <- length(train.execs[[1]]$ref.arg.arrays)
  arg.params <- lapply(1:narg, function(k) {
    if (is.null(train.execs[[1]]$ref.grad.arrays[[k]])) {
      result <- NULL
    } else {
      result <- reduce.sum(lapply(train.execs, function(texec) {
        mx.nd.copyto(texec$ref.arg.arrays[[k]], mx.cpu())
      })) / ndevice
    }
    return(result)
  })
  names(arg.params) <- names(train.execs[[1]]$ref.arg.arrays)
  arg.params <- mx.util.filter.null(arg.params)
  # Get the auxiliary
  naux <- length(train.execs[[1]]$ref.aux.arrays)
  if (naux != 0) {
    aux.params <- lapply(1:naux, function(k) {
      reduce.sum(lapply(train.execs, function(texec) {
        mx.nd.copyto(texec$ref.aux.arrays[[k]], mx.cpu())
      })) / ndevice
    })
    names(aux.params) <- names(train.execs[[1]]$ref.aux.arrays)
  } else {
    aux.params <- list()
  }
  # Get the model
  model <-
    list(symbol = symbol,
         arg.params = arg.params,
         aux.params = aux.params)
  return(structure(model, class = "MXFeedForwardModel"))
}

