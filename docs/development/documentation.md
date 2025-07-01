The documentation has been built using [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) as a static website, and is hosted on the GitHub Pages service.

The doc files are located in the `docs/` directory and configured in the `mkdocs.yml` file.

## Editing

To edit existing or add new pages to the documentation, navigate to the `docs/` directory.

Each page is its own markdown file (.md), and the files have been ordered in subdirectories for a clearer distinction between sections. However, the actual structure of these pages in the documentation is set in the `mkdocs.yml` file under the `nav` keyword.

To add a new page, create your markdown file and link to it in `mkdocs.yml`.

To edit an existing file, simply open it and make your changes. It is recommended to familiarize yourself with the [markdown format](https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/quickstart-for-writing-on-github) beforehand. 

## Updating on GitHub Pages

The file `.github/workflows/ci.yml` has been setup so that upon each push to the repository, the docs pages are updated also on GitHub Pages, without further action required from the developer.

## Viewing locally

To edit and view changes locally, you will need to install the required modules using pip:

```
pip install mkdocs-material
pip install mkdocstrings-python
pip install mkdocs-git-revision-date-localized-plugin
pip install mkdocs-git-authors-plugin
```

The docs can then be launched on a local server using 

```
cd path/to/SOM
python -m mkdocs serve
```

and opened in a web browser at [http://127.0.0.1:8000](http://127.0.0.1:8000).
