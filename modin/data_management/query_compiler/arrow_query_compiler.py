from .pandas_query_compiler import PandasQueryCompiler


class ArrowQueryCompiler(PandasQueryCompiler):

    def query(self, **kwargs):
        """Query columns of the DataManager with a boolean expression.

        Args:
            expr: Boolean expression to query the columns with.

        Returns:
            DataManager containing the rows where the boolean expression is satisfied.
        """
        dtype_obj = self.dtypes
        columns = self.columns

        def query_builder(arrow_table, **kwargs):
            expr = kwargs['expr']
            Expr(expr)
            # Gandiva code execute here
            resolvers = dict(zip(columns, dtype_obj))

        func = self._prepare_method(query_builder, **kwargs)
        new_data = self.map_across_full_axis(1, func)
        # Query removes rows, so we need to update the index
        new_index = self._compute_index(0, new_data, True)
        return self.__constructor__(new_data, new_index, self.columns, self.dtypes)

