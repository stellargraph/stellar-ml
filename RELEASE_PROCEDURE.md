## StellarGraph Library Release Procedure

1. **Create release branch**

   - Create and push the release branch from the latest `develop`
     ```shell
     git fetch
     git checkout -b release-X.X.X origin/develop
     git push -u origin release-X.X.X
     ```

   - Release-related changes are made via Pull Requests from feature branches into the new release branch
     ```shell
     git checkout -b release-X.X.X-changes release-X.X.X
     git push -u origin release-X.X.X-changes
     ```

   - Make the release changes described below.
     - MUST do:
       - Version bumping: Change version from “X.X.Xb” to “X.X.X”. E.g. version=”0.2.0b” to version=”0.2.0”
         - `stellargraph/version.py`
         - `meta.yaml`
       - Update expected versions of demo notebooks: `scripts/format_notebooks.py --default --overwrite demos/`
       - Update Changelog section header and "Full Changelog" link to point to specific version tag instead of `HEAD`. Note: these links will be broken until the tag is pushed later.
     - CAN do:
       - Minor bug fixes if necessary
     - NEVER do:
       - Add new features

   - Commit and push the changes to `release-X.X.X-changes`
     ```shell
     git commit -m "Bump version"
     git push -u origin release-X.X.X-changes
     ```

   - Make a PR from `release-X.X.X-changes` into `release-X.X.X` and merge once approved

   - Once the `release-X.X.X` branch is ready to be merged, create a new Pull Request from the release branch into `master`. This should only be used to exercise CI and for the rest of the team the approve that all necessary changes have been made for release, and **not for doing the actual merge**. **The actual merge into master should be done locally in the next step.**

2. **Merge release branch into `master` locally**

    This step gets your local `master` branch into release-ready state.

    Pull any changes into your local release branch
    ```shell
    git checkout release-X.X.X
    git pull
    ```

    Merge changes into `master`
    ```shell
    git checkout master
    git merge --no-ff release-X.X.X -m "Release X.X.X"
    git tag -a vX.X.X -m "Release X.X.X"
    ```

3. **Upload to PyPI**

    NOTE: An account on PyPI is required to upload - create an account and ask a team member to add you to the organisation.

   - Install build/upload requirements:
     ```shell
     pip install wheel twine
     ```
   - Build distribution files:
     ```shell
     python setup.py sdist bdist_wheel
     ```
     This will create files `stellargraph-<version>.tar.gz` and `stellargraph-<version>-py3-none-any.whl` under the `dist` directory.
   - Upload to PyPi
     ```shell
     twine upload dist/stellargraph-<version>*
     ```
   - Check upload is successful: https://pypi.org/project/stellargraph/

4. **Upload to Conda Cloud**

   NOTE: An account on Conda Cloud is required to upload - create an account and ask a team member to add you to the organization.

   NOTE: These instructions are taken from https://docs.anaconda.com/anaconda-cloud/user-guide/tasks/work-with-packages/)

   - Turn off auto-uploading
     ```shell
     conda config --set anaconda_upload no
     ```
   - Build package
     ```shell
     conda build .
     ```

      NOTE: The Conda package is also built in CI, and uploaded to a Buildkite artifact in the "conda build" stage of the pipeline.  It's possible to download this artifact to be uploaded in the following step, rather than building the conda package locally.

   - Upload to Anaconda Cloud in the “stellargraph” organization
     ```shell
     conda build . --output # find the path to the package
     anaconda login
     anaconda upload -u stellargraph /path/to/conda-package.tar.bz2
     ```

5. **Make release on GitHub**

    After successfully publishing to PyPi and Conda, we now want to make the release on GitHub.

   - Temporarily turn off branch protection on the `master` branch. Ask a team member if you are unsure.
   - Push `master` branch
     ```shell
     git push --follow-tags origin master
     ```
   - Turn branch protection back on.
   - Go to the tags on the GitHub stellargraph homepage: https://github.com/stellargraph/stellargraph/tags
   - Next to the release tag, click the “...” button and select “create release”
   - Add the title and text of the metadata: a title “Release X.X.X” and text copied from the changelog is good practice
   - Click “Publish release”

6. **Get `develop` into correct state for next development version**

    We want the merge any of the changes made during the release back into `develop`, and make sure the new version in `develop` is correct.

   - Switch to `develop` branch:
     ```shell
     git checkout develop
     ```
   - Increase the version: in `stellargraph/version.py`, change version from `X.X.X` to `X.X+1.Xb`. E.g. `__version__ = "0.2.0"` to `__version__ = "0.3.0b"`. (To stay consistent we use `b` to indicate “beta”, but python will accept any string after the number. In semantic versioning: first number for major release, second number for minor release, third number for hotfixes.)
     ```shell
     git add stellargraph/version.py

     # make sure things have the correct format and the notebooks are up to date
     black .
     scripts/format_notebooks.py --default --overwrite demos/

     git commit -m "Bump version"
     ```
   - Merge `master` into `develop` and resolve conflict by using the new version in `develop`:
     ```shell
     git merge master
     ```
   - Temporarily turn off branch protection on the `develop` branch. Ask a team member if you are unsure.
   - Push the merge commit (and the version change):
     ```shell
     git push origin develop
     ```
   - Turn branch protection back on.

7. **Prompt Binder to generate the docker images**

    [Binder](https://mybinder.org) uses a docker image to package up the state of a repository. It takes a long time to build, and is only built lazily, for the first user to click one of our "launch binder" buttons. It is your job to do this:

   1. For the `master` branch:
      - Find [any demo notebook](https://github.com/stellargraph/stellargraph/blob/master/demos/basics/loading-pandas.ipynb) on the `master` branch
      - Click [the "launch binder" button](https://mybinder.org/v2/gh/stellargraph/stellargraph/master?urlpath=lab/tree/demos/basics/loading-pandas.ipynb) (or just click this link)
      - Wait for the "Starting repository: stellargraph/stellargraph" loading screen to switch to a Jupyter environment

   2. For the release tag:
      - Find the [specific release version](https://readthedocs.org/projects/stellargraph/versions/) of documentation just built on readthedocs (eg: `v1.0.0`, not the `stable` or `latest` version).
      - Navigate to any demo notebook in this documentation
      - Click the "launch binder" button
      - Wait for the "Starting repository: stellargraph/stellargraph" loading screen to switch to a Jupyter environment

## More Information

Gitflow Examples:
https://gitversion.readthedocs.io/en/latest/git-branching-strategies/gitflow-examples/
