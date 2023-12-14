class Classifier:
    """
    Wrapping methods common to unsupervised and supervised classifiers
    """

    def __init__(self, categories):
        """
        Builds class for managing functions common to unsupervised and supervised classifiers

        Parameters
        ----------
        categories : List[Category]
            List of Category objects
        """
        self.X = None
        self.categories = sorted(categories, key = lambda c: c['name']) if categories is not None else None
        self.pipeline = None

    def generate_training_matrix_X(self, peakmatrix_df = None):
        """
        Generates training matrix X from PeakMatrix Dataframe.

        Parameters
        -------
        peakmatrix_df : pd.DataFrame, optional
            Data needed to build the training matrix. By default None

        Returns
        -------
        pandas.DataFrame
            A Pandas Dataframe with the training matrix X
        """
        ## Set wavenumber as indexes instead of columns
        peakmatrix_df, _ = self.set_wavenumber_as_dataframe_index(peakmatrix_df)

        ## X: Training matrix
        training_matrix = peakmatrix_df.T
        ## NOTE: scikit.learn feature names only support names that are all strings (v1.2)
        if training_matrix.columns.dtype == 'int64':
            # This is to force 1234 be converted "1234.0", so it could match with other float headers
            training_matrix.columns = training_matrix.columns.astype(float).map(str)
        else:
            training_matrix.columns = training_matrix.columns.map(str)

        # X: Training matrix
        return training_matrix

    def set_wavenumber_as_dataframe_index(self, dataframe):
        """
        Get dataframe Wavenumber column and convert it to the dataframe index. Return the wavenumber list too.

        Parameters
        ----------
        dataframe : pd.DataFrame
            DataFrame expected to contain a column named 'Wavenumber'

        Returns
        -------
        tuple(dataframe, list)
            Same DataFrame without the 'Wavenumber' column. Those values are now set as DataFrame index
            List of wavenumbers
        """

        data_wn = []
        if 'Wavenumber' in dataframe: 
            data_wn = dataframe['Wavenumber'].copy().tolist()
            del dataframe['Wavenumber']
            dataframe.index = data_wn

        return dataframe, data_wn

    def get_samples(self, X = None):
        """
        Returns the sample names linked in matrix X.

        Parameters
        -------
        X : pd.DataFrame, optional
            Training matrix
        
        Returns
        -------
        list 
            Each element is a dict with { 'name': _ }
        """
        samples = list(X.index.values.copy())
        samples = [{'name': sp_id} for sp_id in samples]
        
        return samples