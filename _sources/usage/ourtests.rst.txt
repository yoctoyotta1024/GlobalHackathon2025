Our Tests
=========

Testing Python Code
###################

This project uses pytest for executing the Python tests in the `./tests` directory. These tests are
also included in the CI.yml.

Simply import the code you want to test (e.g. module, function, class etc.)
within a script called ``./tests/test_[name].py`` with a "name" of your choosing in the `./tests`
directory.

Then run pytest on the entire tests directory or on your test. For example, ``pytest ./tests`` would test
every test in the `./tests` directory, whereas ``pytest test_[name].py`` runs just your test.
