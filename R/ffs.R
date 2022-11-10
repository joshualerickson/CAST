#' Forward feature selection
#' @description A simple forward feature selection algorithm
#' @param predictors see \code{\link{train}}
#' @param response see \code{\link{train}}
#' @param method see \code{\link{train}}
#' @param metric see \code{\link{train}}
#' @param maximize see \code{\link{train}}
#' @param globalval Logical. Should models be evaluated based on 'global' performance? See \code{\link{global_validation}}
#' @param withinSE Logical Models are only selected if they are better than the
#' currently best models Standard error
#' @param minVar Numeric. Number of variables to combine for the first selection.
#' See Details.
#' @param trControl see \code{\link{train}}
#' @param tuneLength see \code{\link{train}}
#' @param tuneGrid see \code{\link{train}}
#' @param seed A random number used for model training
#' @param verbose Logical. Should information about the progress be printed?
#' @param ffsParallel Logical. Should chunk-based parallism be used.
#' @param ... arguments passed to the classification or regression routine
#' (such as randomForest).
#' @return A list of class train. Beside of the usual train content
#' the object contains the vector "selectedvars" and "selectedvars_perf"
#' that give the order of the best variables selected as well as their corresponding
#' performance (starting from the first two variables). It also contains "perf_all"
#' that gives the performance of all model runs.
#' @details Models with two predictors are first trained using all possible
#' pairs of predictor variables. The best model of these initial models is kept.
#' On the basis of this best model the predictor variables are iteratively
#' increased and each of the remaining variables is tested for its improvement
#' of the currently best model. The process stops if none of the remaining
#' variables increases the model performance when added to the current best model.
#'
#' The internal cross validation can be run in parallel. See information
#' on parallel processing of carets train functions for details.
#'
#' Using withinSE will favour models with less variables and
#' probably shorten the calculation time
#'
#' Per Default, the ffs starts with all possible 2-pair combinations.
#' minVar allows to start the selection with more than 2 variables, e.g.
#' minVar=3 starts the ffs testing all combinations of 3 (instead of 2) variables
#' first and then increasing the number. This is important for e.g. neural networks
#' that often cannot make sense of only two variables. It is also relevant if
#' it is assumed that the optimal variables can only be found if more than 2
#' are considered at the same time.
#'
#' @note This variable selection is particularly suitable for spatial
#' cross validations where variable selection
#' MUST be based on the performance of the model for predicting new spatial units.
#' See Meyer et al. (2018) and Meyer et al. (2019) for further details.
#'
#' Chuck-based parallelism uses the future package to plan a users parallel options.
#'
#' @author Hanna Meyer
#' @seealso \code{\link{train}},\code{\link{bss}},
#' \code{\link{trainControl}},\code{\link{CreateSpacetimeFolds}},\code{\link{nndm}}
#' @references
#' \itemize{
#' \item Gasch, C.K., Hengl, T., Gräler, B., Meyer, H., Magney, T., Brown, D.J. (2015): Spatio-temporal interpolation of soil water, temperature, and electrical conductivity in 3D+T: the Cook Agronomy Farm data set. Spatial Statistics 14: 70-90.
#' \item Meyer, H., Reudenbach, C., Hengl, T., Katurji, M., Nauß, T. (2018): Improving performance of spatio-temporal machine learning models using forward feature selection and target-oriented validation. Environmental Modelling & Software 101: 1-9.  \doi{10.1016/j.envsoft.2017.12.001}
#' \item Meyer, H., Reudenbach, C., Wöllauer, S., Nauss, T. (2019): Importance of spatial predictor variable selection in machine learning applications - Moving from data reproduction to spatial prediction. Ecological Modelling. 411, 108815. \doi{10.1016/j.ecolmodel.2019.108815}
#' }
#' @examples
#' \dontrun{
#' data(iris)
#' ffsmodel <- ffs(iris[,1:4],iris$Species)
#' ffsmodel$selectedvars
#' ffsmodel$selectedvars_perf
#'}
#'
#' # or perform model with target-oriented validation (LLO CV)
#' #the example is described in Gasch et al. (2015). The ffs approach for this dataset is described in
#' #Meyer et al. (2018). Due to high computation time needed, only a small and thus not robust example
#' #is shown here.
#'
#' \dontrun{
#' #run the model on three cores:
#' library(doParallel)
#' cl <- makeCluster(3)
#' registerDoParallel(cl)
#'
#' #load and prepare dataset:
#' dat <- get(load(system.file("extdata","Cookfarm.RData",package="CAST")))
#' trainDat <- dat[dat$altitude==-0.3&year(dat$Date)==2012&week(dat$Date)%in%c(13:14),]
#'
#' #visualize dataset:
#' ggplot(data = trainDat, aes(x=Date, y=VW)) + geom_line(aes(colour=SOURCEID))
#'
#' #create folds for Leave Location Out Cross Validation:
#' set.seed(10)
#' indices <- CreateSpacetimeFolds(trainDat,spacevar = "SOURCEID",k=3)
#' ctrl <- trainControl(method="cv",index = indices$index)
#'
#' #define potential predictors:
#' predictors <- c("DEM","TWI","BLD","Precip_cum","cday","MaxT_wrcc",
#' "Precip_wrcc","NDRE.M","Bt","MinT_wrcc","Northing","Easting")
#'
#' #run ffs model with Leave Location out CV
#' set.seed(10)
#' ffsmodel <- ffs(trainDat[,predictors],trainDat$VW,method="rf",
#' tuneLength=1,trControl=ctrl)
#' ffsmodel
#'
#' #compare to model without ffs:
#' model <- train(trainDat[,predictors],trainDat$VW,method="rf",
#' tuneLength=1, trControl=ctrl)
#' model
#' stopCluster(cl)
#'}
#' @export ffs
#' @aliases ffs
#' @importFrom dplyr '%>%'

ffs <- function (predictors,
                 response,
                 method = "rf",
                 metric = ifelse(is.factor(response), "Accuracy", "RMSE"),
                 maximize = ifelse(metric == "RMSE", FALSE, TRUE),
                 globalval=FALSE,
                 withinSE = FALSE,
                 minVar = 2,
                 trControl = caret::trainControl(),
                 tuneLength = 3,
                 tuneGrid = NULL,
                 seed = sample(1:1000, 1),
                 verbose=TRUE,
                 ffsParallel = FALSE,
                 ...){

  trControl$returnResamp <- "final"
  trControl$savePredictions <- "final"

  if(inherits(response,"character")){
    response <- factor(response)
    if(metric=="RMSE"){
      metric <- "Accuracy"
      maximize <- TRUE
    }
  }
  if (trControl$method=="LOOCV"){
    if (withinSE==TRUE){
      print("warning: withinSE is set to FALSE as no SE can be calculated using method LOOCV")
      withinSE <- FALSE
    }}

  if(globalval){
    if (withinSE==TRUE){
      print("warning: withinSE is set to FALSE as no SE can be calculated using global validation")
      withinSE <- FALSE
    }}

  #### chose initial best model from all combinations of two variables
  minGrid <- t(data.frame(combn(names(predictors),minVar)))

  #### Run the initial models based on minGrid
  if(isTRUE(ffsParallel)){

    initial_models <-
      furrr::future_map(split(minGrid, 1:nrow(minGrid)),~chunk_model(.,
                                                                            predictors = predictors,
                                                                            response = response,
                                                                            method = method,
                                                                            metric = metric,
                                                                            maximize = maximize,
                                                                            globalval = globalval,
                                                                            minVar = minVar,
                                                                            trControl = trControl,
                                                                            tuneLength = tuneLength,
                                                                            tuneGrid = tuneGrid,
                                                                            seed = seed,
                                                                            verbose = verbose))
  } else {

    initial_models <-
      purrr::map(split(minGrid, 1:nrow(minGrid)),~chunk_model(.,
                                                                     predictors = predictors,
                                                                     response = response,
                                                                     method = method,
                                                                     metric = metric,
                                                                     maximize = maximize,
                                                                     globalval = globalval,
                                                                     minVar = minVar,
                                                                     trControl = trControl,
                                                                     tuneLength = tuneLength,
                                                                     tuneGrid = tuneGrid,
                                                                     seed = seed,
                                                                     verbose = verbose
      ))

  }

  # bind the list of performance stats and vars per model
  initial_models <- plyr::rbind.fill(initial_models)

  best_model <- model_results(initial_models, maximize)

  #### increase the number of predictors by one (try all combinations)
  # and test if model performance increases

  if(verbose){
    print(paste0(paste0("vars selected: ",paste(best_model$vars, collapse = ',')),
                 " with ",metric," ",round(best_model$actmodelperf,3)))
  }

  for (k in 1:(length(names(predictors))-minVar)){

    startvars <- unlist(strsplit(best_model$vars, ','))

    nextvars <- names(predictors)[-which(
      names(predictors)%in%startvars)]

    minGrid <- t(data.frame(combn(c(startvars, nextvars),length(startvars)+1)))[1:length(nextvars),]

    if (length(startvars)<(k+(minVar-1))){
      message(paste0("Note: No increase in performance found using more than ",
                     length(startvars), " variables"))

      return(overall_models)
      break()
    }

    if(isTRUE(ffsParallel)){

      model <-
        furrr::future_map(split(minGrid, 1:nrow(minGrid)),~chunk_model(.,
                                                                       predictors = predictors,
                                                                       response = response,
                                                                       method = method,
                                                                       metric = metric,
                                                                       maximize = maximize,
                                                                       globalval = globalval,
                                                                       minVar = minVar,
                                                                       trControl = trControl,
                                                                       tuneLength = tuneLength,
                                                                       tuneGrid = tuneGrid,
                                                                       seed = seed,
                                                                       verbose = verbose))
    } else {

      model <-
        purrr::map(split(minGrid, 1:nrow(minGrid)),~chunk_model(.,
                                                                predictors = predictors,
                                                                response = response,
                                                                method = method,
                                                                metric = metric,
                                                                maximize = maximize,
                                                                globalval = globalval,
                                                                minVar = minVar,
                                                                trControl = trControl,
                                                                tuneLength = tuneLength,
                                                                tuneGrid = tuneGrid,
                                                                seed = seed,
                                                                verbose = verbose
        ))

    }

    # bind the list of performance stats and vars per model
    model <- plyr::rbind.fill(model)
    model$var_number <- length(startvars+1)

    if(k == 1){
      overall_models <- rbind(initial_models, model)

    } else {

      overall_models <- rbind(overall_models, model)
    }

    new_model <- model_results(model, maximize)

    if(isBetter(new_model$actmodelperf,best_model$actmodelperf,
                best_model$actmodelperfSE, #SE from model with nvar-1
                maximization=maximize,withinSE=withinSE)){
      best_model <- new_model
    }

  }

  return(overall_models)

}



#' Chunk-based Initial Model Function
#'
#' @param minGrid
#' @param predictors
#' @param response
#' @param method
#' @param metric
#' @param maximize
#' @param globalval
#' @param minVar
#' @param trControl
#' @param tuneLength
#' @param tuneGrid
#' @param seed
#' @param verbose
#' @param ...
#' @noRd
#'
#' @return A data.frame with performance statistics and variable pairs
#'
chunk_model <- function(minGrid,
                               predictors,
                               response,
                               method = 'rf',
                               metric = ifelse(is.factor(response), "Accuracy", "RMSE"),
                               maximize = ifelse(metric == "RMSE", FALSE, TRUE),
                               globalval=FALSE,
                               minVar = 2,
                               trControl = caret::trainControl(),
                               tuneLength = 3,
                               tuneGrid = NULL,
                               seed = sample(1:1000, 1),
                               verbose=TRUE,
                               ...) {

  set.seed(seed)
  #adaptations for pls:
  if(method=="pls"&!is.null(tuneGrid)&any(tuneGrid$ncomp>minVar)){
    tuneGrid <- data.frame(ncomp=tuneGrid[tuneGrid$ncomp<=minVar,])
    if(verbose){
      print(paste0("note: maximum ncomp is ", minVar))
    }
  }
  #adaptations for tuning of ranger:
  if(method=="ranger"&!is.null(tuneGrid)&any(tuneGrid$mtry>minVar)){
    tuneGrid$mtry <- minVar
    if(verbose){
      print("invalid value for mtry. Reset to valid range.")
    }
  }

  #train model:
  model <- caret::train(predictors[,minGrid],
                        response,
                        method=method,
                        metric=metric,
                        trControl=trControl,
                        tuneLength = tuneLength,
                        tuneGrid = tuneGrid,
                        ...)

  ### compare the model with the currently best model
  if (globalval){
    perf_stats <- global_validation(model)[names(global_validation(model))==metric]
  }else{
    perf_stats <- model$results[,names(model$results)==metric]
  }

  actmodelperf <- evalfunc(maximize, perf_stats)
  actmodelperfSE <- se(
    sapply(unique(model$resample$Resample),
           FUN=function(x){mean(model$resample[model$resample$Resample==x,
                                               metric],na.rm=TRUE)}))

  variablenames <- names(model$trainingData)[-length(names(model$trainingData))]

  final_stats <- cbind.data.frame(actmodelperf, actmodelperfSE, vars = toString(variablenames))

}

#' Maximize Helper
#'
#' @param maximize Logical
#' @param value Numeric
#'
#' @return A function.
#' @noRd

evalfunc <- function(maximize, value){

if(maximize) ev <- function(x){max(x,na.rm=TRUE)}
if(!maximize) ev <- function(x){min(x,na.rm=TRUE)}

      ev(value)
}


#' Standard Error
#'
#' @param x numeric
#'
#' @return A numeric vector.
#' @noRd

se <- function(x){sd(x, na.rm = TRUE)/sqrt(length(na.exclude(x)))}

#' Comparing Model Helper
#'
#' @param actmodelperf
#' @param bestmodelperf
#' @param bestmodelperfSE
#' @param maximization
#' @param withinSE
#'
#' @return A model.
#' @noRd
isBetter <- function (actmodelperf,bestmodelperf,
                      bestmodelperfSE=NULL,
                      maximization=FALSE,
                      withinSE=FALSE){
  if(withinSE){
    result <- ifelse (!maximization, actmodelperf < bestmodelperf-bestmodelperfSE,
                      actmodelperf > bestmodelperf+bestmodelperfSE)
  }else{
    result <- ifelse (!maximization, actmodelperf < bestmodelperf,
                      actmodelperf > bestmodelperf)
  }
  return(result)
}


#' Model Result Helper
#'
#' @param model A data.frame
#' @param maximize Logical
#'
#' @return A data.frame.
#' @noRd

model_results <- function(model, maximize) {

  if(isTRUE(maximize)){

  selectedvars <- unlist(strsplit(model[max(model$actmodelperf) == model$actmodelperf,]$vars, ', '))
  selectedvars_perf <- model[max(model$actmodelperf) == model$actmodelperf,]$actmodelperf
  selectedvars_SE <- model[model$actmodelperf == selectedvars_perf,]$actmodelperfSE

  data.frame(actmodelperf = selectedvars_perf,
             actmodelperfSE = selectedvars_SE,
             vars = toString(selectedvars))

  } else {

  selectedvars <- unlist(strsplit(model[min(model$actmodelperf) == model$actmodelperf,]$vars, ', '))
  selectedvars_perf <- model[min(model$actmodelperf) == model$actmodelperf,]$actmodelperf
  selectedvars_SE <- model[model$actmodelperf == selectedvars_perf,]$actmodelperfSE

  data.frame(actmodelperf = selectedvars_perf,
             actmodelperfSE = selectedvars_SE,
             vars = toString(selectedvars))

  }

}
