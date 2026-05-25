"""PySpark 3.5 + Delta Lake medallion ETL.

Submodules import ``pyspark`` lazily so unit tests can exercise pure-Python
helpers (e.g. :class:`gamereco.etl.splits.SplitFractions`) without requiring
a Spark install.
"""
