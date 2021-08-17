Installation
=====

.. _installation:

Installation
------------

To use AML, first install it from git repository:

.. code-block:: console

   pip install -e .

For tests first install Tox:

.. code-block:: console

    pip install tox
    # and then execute:
    tox -e test_pipeline

To build documentation:

.. code-block:: console

    cd docs
    make html