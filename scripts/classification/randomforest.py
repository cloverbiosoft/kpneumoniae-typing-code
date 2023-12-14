from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline

from .supervisedclassifier import SupervisedClassifier

class RandomForestAnalysis(SupervisedClassifier):
    def __init__(self, categories, n_estimators = 100, max_features = 'sqrt', max_depth = None, min_samples_split = 2, 
        min_samples_leaf = 1):
        """
        Performs Random Forest analysis algorithm (RF)

        Parameters
        ----------
        categories : List[dict]
            The actual category for each sample in the PeakMatrix
        n_estimators : int, optional
            The number of trees in the forest, by default 100
        max_features : {"sqrt", "log2"} or int, optional
            The number of maximum features to consider at every split, by default sqrt
        max_depth : int or None, optional
            The number of maximum levels at each tree, by default None
        min_samples_split : int, optional
            The number of minimum samples required to split a node, by default 2
        min_samples_leaf : int, optional
            The number of minimum samples required at each leaf, by default 1
        """
        super().__init__(categories)
        self.n_estimators = n_estimators if n_estimators is not None else 100
        self.max_features = max_features if max_features is not None else 'sqrt'
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split if min_samples_split is not None else 2
        self.min_samples_leaf = min_samples_leaf if min_samples_leaf is not None else 1

    def _init_X_Y(self, peakmatrix_df = None):
        """
        Builds X (training data) and Y (target categories)

        Parameters
        ----------
        peakmatrix_df : pd.DataFrame, optional
            Data needed to build the training matrix. By default None
        """
        self.X = self.generate_training_matrix_X(peakmatrix_df = peakmatrix_df)
        self.Y = self.generate_target_matrix_Y(self.X)

    def fit_model(self, n_estimators = 100, max_features='sqrt', max_depth=None, min_samples_split = 2, min_samples_leaf = 1):
        """
        Fit model to available data

        Parameters
        ----------
        n_estimators : int, optional
            The number of trees in the forest, by default 100
        max_features : {"sqrt", "log2"} or int, optional
            The number of maximum features to consider at every split, by default sqrt
        max_depth : int or None, optional
            The number of maximum levels at each tree, by default None
        min_samples_split : int, optional
            The number of minimum samples required to split a node, by default 2
        min_samples_leaf : int, optional
            The number of minimum samples required at each leaf, by default 1
        """
        pipeline = self.get_analysis_model({
            'n_estimators': n_estimators,
            'max_features': max_features,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf})

        pipeline.fit(self.X, self.Y)

        return pipeline
        
    def run(self, peakmatrix_df = None):
        """
        Runs the full process (fit and transform)

        Parameters
        ----------
        peakmatrix_df : pd.DataFrame, optional
            The dataset to be fitted, by default None

        Returns
        -------
        dict
            Returns the leave-one-out cross validation
        """

        ## Initialize self.X and self.Y
        self._init_X_Y(peakmatrix_df = peakmatrix_df)

        ## Fit the model
        self.pipeline = self.fit_model(n_estimators = self.n_estimators, max_features = self.max_features, max_depth = self.max_depth, 
            min_samples_split = self.min_samples_split, min_samples_leaf = self.min_samples_leaf)
   
        ## Cross validation
        confusion_matrix_kfold = self.loo_cross_validation()
        result = {"confusion_matrix_kfold": confusion_matrix_kfold}

        return result

    def loo_cross_validation(self):
        """
        Performs RandomForestAnalysis by applying a Leave-One-Out cross validation

        Returns
        -------
        Confussion matrix
        """

        print("Performing the cross validation")

        # Initialize classifier_params with main params from classifier
        classifier_params = {
            'n_estimators': self.n_estimators,
            'max_features': self.max_features,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf}
        
        return self.loo_cross_validation_classifier(self.X, classifier_params)

    def get_analysis_model(self, classifier_params):
        """
        Get RandomForestClassifier initialization

        Parameters
        ----------
        classifier_params : dict
            Dict with all params for classifier initialization

        Returns
        -------
        Pipeline
        """
        ## Initialize steps list for Pipeline
        steps = []

        ## The only step is the RandomForestClassifier itself
        steps.append(('rf', 
            RandomForestClassifier(warm_start = True, oob_score = True,
                n_estimators = classifier_params['n_estimators'],
                max_features = classifier_params['max_features'],
                max_depth = classifier_params['max_depth'],
                min_samples_split = classifier_params['min_samples_split'],
                min_samples_leaf = classifier_params['min_samples_leaf'],
                random_state=42))
        )

        return Pipeline(steps)

    def get_confusion_matrix(self, X, classifier = None):
        """
        Performs prediction of X and returns confusion matrix from prediction data returned by RandomForestAnalysis
        Parameters
        ----------
        X : pd.DataFrame
            Dataset to be predicted
        classifier : RandomForestAnalysis, by default None

        Returns
        -------
        sklearn `confusion matrix`
        """
        # Check classifier
        classifier = self.pipeline if classifier is None else classifier
        
        # Perform prediction if model is fitted
        try:
            prediction = classifier.predict(X)
        except NotFittedError as e:
            raise Exception('PLS Model not fitted')
        except:
            # if here, classifier is None
            raise Exception('Classifier is None')
        
        return self.build_confusion_matrix(X, prediction)
