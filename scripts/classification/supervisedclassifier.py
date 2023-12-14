import abc
import math
import numpy as np
from scripts.classification.classifier import Classifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

class SupervisedClassifier(Classifier, metaclass=abc.ABCMeta):

    def __init__(self, categories):
        """
        Class for managing supervised classifiers

        Parameters
        ----------
        categories : List[dict]
            The actual category for each sample in the PeakMatrix
        """
        super().__init__(categories)
        self.Y = None

    @abc.abstractmethod
    def get_analysis_model(self, classifier_params):
        return
    
    @abc.abstractmethod
    def get_confusion_matrix(X_test, classifier):
        return

    def generate_target_matrix_Y(self, X=None):
        """
        Generates target matrix Y from categories content, aligning values with X matrix

        Parameters
        -------
        X : pd.DataFrame, optional
            Training matrix, by default None

        Returns
        ------
        list
            Category score for each element of the training matrix
        """

        # Check categories
        if self.categories is None:
            raise Exception('CategoriesNotProvided')

        # Check training matrix
        if X is None and self.X is None:
            raise Exception('TrainingMatrixXNotDefined')

        elif X is None:
            X = self.X

        # Map categories by sample
        categories_T = {}
        for category in self.categories:
            for sample_id in category['samples']:
                categories_T[sample_id] = category['score']
                
        # Y: Targets list
        # Save in Y category value for each sample/column
        Y = [categories_T[c] for c in X.index]
        
        return Y
    
    def build_confusion_matrix(self, X, prediction):
        """
        Returns sklearn `confusion matrix` from `prediction` data for `X` matrix

        Parameters
        -------
        X : pd.DataFrame
            Training matrix
        prediction : list
            Values of prediction for each sample

        Returns
        -------
        ndarray of shape (n_categories, n_categories)
            Confusion matrix from sklearn library
        """
        known_y = []
        predicted_y = []
        samples = self.get_samples(X)

        ## Add idx as unique identifier for each category
        for idx, category in enumerate(self.categories):
            category['idx'] = idx
            
        for (idx, sample) in enumerate(samples):
            predicted_score = prediction[idx]
            predicted_category = self._get_predicted_category(predicted_score)
            known_category = next((c for c in self.categories if sample['name'] in c['samples']), None)
            predicted_y.append(predicted_category['idx'])
            known_y.append(known_category['idx'])

        return self._get_confusion_matrix(known_y, predicted_y)

    def _get_confusion_matrix(self, y_true, y_predicted):
        """
        Returns confusion matrix data according to the actual and predicted labels provided

        Parameters
        ----------
        y_true : array-like of shape (n_samples,)
            The truth labels
        y_predicted : array-like of shape (n_samples,)
            The predicted labels

        Returns
        -------
        dict
            The content of the confusion matrix and the categories in the same order that the one used in the confussion matrix
        """
        matrix = {}
        matrix['data'] = confusion_matrix(y_true, y_predicted, labels=[c['idx'] for c in self.categories])
        matrix['columns'] = self.categories
        return matrix

    def loo_cross_validation_classifier(self, X, classifier_params = None):
        """
        Performs classifier analysis by applying a Leave-One-Out cross validation

        Parameters
        ----------
        X : pd.DataFrame, optional
            The dataset to be fitted, by default None
        classifier_params : dict, optional
            Dict with main params for classifier initialization, by default None
        
        Returns
        -------
        Confussion matrix 
        """
        
        # Initialize variables
        confusion_matrix_acc = {'data': None, 'columns': None}

        # Generates target matrix Y
        y = np.array(self.generate_target_matrix_Y(X=X))

        # As leave-one-out case, n_splits must be the number of samples
        n_splits = X.shape[0]

        # Apply k-fold cross validation by splitting X matrix in X_test and X_train
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=np.random.RandomState(324546))

        kindex = 1
        for train_index, test_index in kf.split(X, y):
            # Split train and test data for this kfold iteration
            X_train = X.iloc[train_index, : ]
            X_test = X.iloc[test_index, : ]

            ## ---- For confusion matrix
            # Fit classifier model with train data for obtaining confusion matrix
            Y_train_for_confusion_matrix = y[train_index]
            classifier_confusion_matrix = self._get_fit_classifier(X_train, Y_train_for_confusion_matrix, classifier_params)

            # Get confusion matrix for test data
            print("Obtaining the k-fold confusion matrix {0}/{1}".format(kindex, n_splits))
            confusion_matrix = self.get_confusion_matrix(X_test, classifier_confusion_matrix)
            kindex += 1
            
            # Aggregate result to a common confusion matrix
            confusion_matrix_acc['data'] = confusion_matrix['data'] if confusion_matrix_acc['data'] is None else (confusion_matrix_acc['data'] + confusion_matrix['data'])
            confusion_matrix_acc['columns'] = confusion_matrix['columns']

        return confusion_matrix_acc
       
    def _get_fit_classifier(self, X_train, y_train, classifier_params):
        """
        Returns a classifier fitted for X_train data

        Parameters
        ----------
        X_train : pd.DataFrame of shape (n_samples, n_features)
            Training data
        y_train : array-like of shape (n_samples,)
            category labels for the training data
        classifier_params : dict
            extra parameters for building the classifier

        Returns
        -------
        fitted sklearn classifier
        """
        # Get model for train data
        classifier_params['n_samples'] = X_train.shape[0]
        classifier = self.get_analysis_model(classifier_params)
        
        # Create pipeline model - use pipeline steps from get_analysis_model output
        pipeline = Pipeline(steps=classifier.steps)
        
        pipeline.fit(X_train, y_train)

        return pipeline

    def _get_predicted_category(self, prediction_score):
        """
        Returns the category that is closer to the `prediction_score` value
        
        Parameters
        -------
        prediction_score : number
            Prediction score value assigned to a sample 

        Returns
        -------
        category
            Category predicted to the prediction score value
        """
        predicted_category = None
        diff = math.inf
        for category in self.categories:
            distance = abs(category['score'] - prediction_score)
            if(distance < diff):
                diff = distance
                predicted_category = category
      
        return predicted_category
