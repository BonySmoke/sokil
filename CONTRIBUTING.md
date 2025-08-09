# Contributing Guidelines

We are grateful for your willingness to contribute to this project! We are interested in any features, bug fixes, new usage examples, etc.

## How to Contribute

1. Fork this repository, develop, and test your changes.
2. Submit a pull request.
3. Make sure all GitHub actions pass successfully.

***NOTE***: In order to make testing and merging of PRs easier, please submit changes for different fixes/features/improvements in separate PRs.

### Technical Requirements

* Must pass the CI job for linting. Please run `make lint` in the root of the project to know if the project complies with the requirements.
* New code must be covered by unit tests and pass the corresponding CI job. Please run `make test-unit` in the root of the project.

Once changes have been merged, the release will be created by the repository maintainers.

## 🛠 Development

You can easily run the project by following these steps:

1) Init the project and make changes. The project depends on [uv](https://docs.astral.sh/uv/getting-started/installation/); therefore, please make sure to install it first

```bash
make init-project
```

2) If you need to annotate data for model training, spin up a CVAT instance:

```bash
make cvat-up
make cvat-superuser # create a super user if it's the first time starting CVAT
```

Refer to (docs/training.md)[docs/training.md] for more details.

3) If you need to run a review of a video, run

```
make review
```
