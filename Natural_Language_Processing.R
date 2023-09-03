library(data.table)
library(tidyverse)
library(text2vec)
library(caTools)
library(glmnet)
library(inspectdf)


# 1. Import emails dataset and get familiarized with it. ----

df <- read_csv("emails.csv")

df %>% colnames()
df %>% glimpse()
df %>% inspect_na()

df$spam %>% data.table() %>% table()


# 2. Add 'id' column defined by number of rows.

df$id <- seq_len(nrow(df))
df$id <- df$id %>% as.character()


# 3. Prepare data for fitting to the model. ----

set.seed(123)
split <- df$spam %>% sample.split(SplitRatio = 0.80)
train <- df %>% subset(split == T)
test <- df %>% subset(split == F)

it_train <- train$text %>% 
  itoken(preprocessor = tolower, 
         tokenizer = word_tokenizer,
         ids = train$id,
         progressbar = F) 


vocab <- it_train %>% create_vocabulary()

vocab %>% 
  arrange(desc(term_count)) %>% 
  head(110) %>% 
  tail(10) 

vectorizer <- vocab %>% vocab_vectorizer()
dtm_train <- it_train %>% create_dtm(vectorizer)

dtm_train %>% dim()
identical(rownames(dtm_train), train$id)


# 4. Use cv.glmnet for modeling. ----

glmnet_classifier <- dtm_train %>% 
  cv.glmnet(y = train[['spam']],
            family = 'binomial', 
            type.measure = "auc",
            # nfolds = 10,
            thresh = 0.001,           
            maxit = 1000)             

glmnet_classifier$cvm %>% max() %>% round(3) %>% paste("-> Max AUC")


it_test <- test$text %>% tolower() %>% word_tokenizer()

it_test <- it_test %>% 
  itoken(ids = test$id,
         progressbar = F)

dtm_test <- it_test %>% create_dtm(vectorizer)

preds <- predict(glmnet_classifier, dtm_test, type = 'response')[, 1]
glmnet:::auc(test$spam, preds) %>% round(2)



# 5. Give interpretation for train and test results. ----


# Extract coefficients from the trained glmnet model
coefficients <- coef(glmnet_classifier)

# Identify important terms with non-zero coefficients
important_terms <- vocab$term[which(coefficients != 0)]

# Create a data frame to store important terms and their corresponding coefficients
important_terms_coeff <- data.frame(Term = important_terms, Coefficient = coefficients[coefficients != 0])

# Display the data frame containing important terms and coefficients
important_terms_coeff %>% view()


