
def fit(self, X_train, y_train, X_test=None, y_test=None):
     results = []
      final_pipes = self._make_aml_combinations(
           self.pipeline, self.param_grid)
       for f in final_pipes:
            # this
            amlcv = GridSearchCV(f, cv=3, param_grid={})
            amlcv.fit(X_train, y_train)
            # # or this
            # f.fit(X_train, y_train)
            y_pred_train = amlcv.predict(X_train)
            if X_test is not None:
                y_pred_test = amlcv.predict(X_test)
            letters = string.ascii_lowercase
            pipe_name = ''.join(random.choice(letters) for i in range(10))
            error_train = self.scoring(y_train, y_pred_train)
            if X_test is not None:
                error_test = self.scoring(y_test, y_pred_test)
            else:
                error_test = np.nan
            res = {'name': pipe_name,
                   'params': amlcv.estimator.named_steps,
                   'error_train': round(error_train, 2),
                   'error_test': round(error_test, 2),
                   'train_test_dif': round(error_test / error_train, 2),
                   }
            results.append(res)
        results = pd.DataFrame.from_dict(results)
        return results
