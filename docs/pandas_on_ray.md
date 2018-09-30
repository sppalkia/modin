Pandas on Ray
=============

Pandas on Ray is an early stage DataFrame library that wraps Pandas and
transparently distributes the data and computation. The user does not
need to know how many cores their system has, nor do they need to
specify how to distribute the data. In fact, users can continue using
their previous Pandas notebooks while experiencing a considerable
speedup from Pandas on Ray, even on a single machine. Only a
modification of the import statement is needed, as we demonstrate below.
Once you’ve changed your import statement, you’re ready to use Pandas on
Ray just like you would Pandas.

``` {.sourceCode .python}
# import pandas as pd
import modin.pandas as pd
```

Currently, we have part of the Pandas API implemented and are working
toward full functional parity with Pandas.

Using Pandas on Ray on a Single Node
------------------------------------

In order to use the most up-to-date version of Pandas on Ray, please
follow the instructions on the [installation
page](http://modin.readthedocs.io/en/latest/installation.html)

Once you import the library, you should see something similar to the
following output:

``` {.sourceCode .text}
>>> import modin.pandas as pd

Waiting for redis server at 127.0.0.1:14618 to respond...
Waiting for redis server at 127.0.0.1:31410 to respond...
Starting local scheduler with the following resources: {'CPU': 4, 'GPU': 0}.

======================================================================
View the web UI at http://localhost:8889/notebooks/ray_ui36796.ipynb?token=ac25867d62c4ae87941bc5a0ecd5f517dbf80bd8e9b04218
======================================================================
```

Once you have executed `import modin.pandas as pd`, you're ready to
begin running your pandas pipeline as you were before.

APIs Supported
--------------

Please note, the API is not yet complete. For some methods, you may see
the following:

``` {.sourceCode .text}
NotImplementedError: To contribute to Pandas on Ray, please visit github.com/modin-project/modin.
```

We have compiled a list of currently supported methods
[here](http://modin.readthedocs.io/en/latest/pandas_supported.html).

If you would like to request a particular method be implemented, feel
free to [open an issue](http://github.com/modin-project/modin/issues).
Before you open an issue please make sure that someone else has not
already requested that functionality.

Using Pandas on Ray on a Cluster
--------------------------------

Currently, we do not yet support running Pandas on Ray on a cluster.
Coming Soon!

Examples
--------

You can find an example on our recent [blog
post](http://rise.cs.berkeley.edu/blog/pandas-on-ray) or on the [Jupyter
Notebook](http://gist.github.com/devin-petersohn/f424d9fb5579a96507c709a36d487f24#file-pandas_on_ray_blog_post_0-ipynb)
that we used to create the blog post.
