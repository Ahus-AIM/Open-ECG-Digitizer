# CHANGELOG

<!-- version list -->

## v1.9.3 (2026-06-04)

### Bug Fixes

- Typechecking
  ([`69d68c1`](https://github.com/Ahus-AIM/Open-ECG-Digitizer/commit/69d68c19335ecce4423e457e17c3a16d3d29ebbe))

### Chores

- Add license
  ([`9e712fe`](https://github.com/Ahus-AIM/Open-ECG-Digitizer/commit/9e712feac158d127843faac7d5e100ced00d0e6e))

### Continuous Integration

- Use python3 in release workflows
  ([`04e020b`](https://github.com/Ahus-AIM/Open-ECG-Digitizer/commit/04e020b1778e6c8b610686ae4b11f430cb092720))

### Documentation

- Add preprint to readme ([#37](https://github.com/Ahus-AIM/Open-ECG-Digitizer/pull/37),
  [`cca543c`](https://github.com/Ahus-AIM/Open-ECG-Digitizer/commit/cca543cf0542240a6b5b027510a50ff2a8b38383))

- Update readme with dataset link and citation
  ([#40](https://github.com/Ahus-AIM/Open-ECG-Digitizer/pull/40),
  [`963387f`](https://github.com/Ahus-AIM/Open-ECG-Digitizer/commit/963387ff5abdfa3db91c15ac52d6cf1214345a6a))


## v1.9.2 (2025-10-20)

### Bug Fixes

- Remove non-model code
  ([`c1b41e0`](https://github.com/Ahus-AIM/Open-ECG-Digitizer/commit/c1b41e0bf56e59167e7ba9a03977e8fae3ec886a))

### Documentation

- Add module details to readme
  ([`2c43b09`](https://github.com/Ahus-AIM/Open-ECG-Digitizer/commit/2c43b09646ec1fcb06a095d40438952f7843134e))

- Specify output units
  ([`316f71d`](https://github.com/Ahus-AIM/Open-ECG-Digitizer/commit/316f71ded52d370e9bf6e7e4c40cbd6f57076c37))


## v1.9.1 (2025-10-03)

### Bug Fixes

- Typo in readme
  ([`1c052ff`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/1c052ff052eb41a8a611f2c4ff448d36460a411a))


## v1.9.0 (2025-10-03)

### Documentation

- Update readme to describe overriding config in inference
  ([`2b02686`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/2b026865283e59b4e36a35788494a1b31650e303))

### Features

- Enable overriding of config, remove default config
  ([`428f8b4`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/428f8b4363b2bbe3d72650a6e66733c7daa5c069))


## v1.8.0 (2025-10-02)

### Documentation

- Improve README.md
  ([`32972cf`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/32972cfe3558c6b6c81bcae386e7e8402b19c239))

### Features

- Save metadata along digitized timeseries
  ([`33d349f`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/33d349f3fec80d40755cdf8fff70f8777ca80e40))

- Support variable threshold for crop left/right
  ([`ff49836`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/ff49836d269fcaa2083f90344a8c442de2eac698))


## v1.7.0 (2025-09-17)

### Build System

- Unpin requirements
  ([`4ab7bcf`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/4ab7bcf7e2a520371c874d180049cbc28c1ffa40))

### Code Style

- Change type annotations to make mypy happy
  ([`415038b`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/415038b1a4a28a91f6282addbac09f6ac6673f75))

### Features

- Add config file for gm2024 test set
  ([`8f610ca`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/8f610ca6a6b2dc7950d68deeb07a58711bdf5107))

- Include image aspect for line count estimation
  ([`7cc0a9c`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/7cc0a9caa6017c5a0be740c17326f8f78be7f300))

- Report nan count
  ([`9923d53`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/9923d53f89846b4bd6a8ccce8814d63c93746684))


## v1.6.0 (2025-07-11)

### Bug Fixes

- Apply windowing to reduce effects from edges
  ([`7d57657`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/7d57657712e1c6d6037b99b58fc154b0f0358909))

- Debug false as default, and fix edge case errors
  ([`265bc39`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/265bc39109a0e1a0a7c83ed3ddeddb09a1b1be16))

- Do not step lr scheduler in eval mode
  ([`d5f3632`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/d5f36321133524059259d703a68ff52ce19b383e))

- Set alpha factor as argument
  ([`daf451b`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/daf451b9c4b5b5906b4a61cb4a82fd3073e084c5))

- Update config files
  ([`28673c5`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/28673c50dc80419af57475bc24ddfde321bb7173))

- Update weight path for inference
  ([`42d94df`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/42d94dfef393bd5ca6db8691aa2cd1de2e2d0eca))

### Features

- Add code to calculate metrics
  ([`5686d24`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/5686d24a0be0d40d9c10ffb4d2f3c9874c6c5a43))

- Inference wrapper
  ([`844667e`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/844667e84169d019e949f1e3e8d764084a541551))

- Layout autodetection
  ([`9daf923`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/9daf9233ff7e4634757603085edc3412141d138e))

- Make dewarping optional and change return types
  ([`cfc110d`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/cfc110d480254796df024c9d130400501f0e7c96))

- Update digitizer to save csv files
  ([`1bcb178`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/1bcb1786b7582188a6c7578b7088c0120aa99db1))


## v1.5.0 (2025-07-01)

### Bug Fixes

- Ensure no crash if dewarping gets singular matrix
  ([`a102dd5`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/a102dd57d865c6334d6bed690efab467f54314bd))

### Chores

- Add formatting, linting, and type-check script
  ([`34b4f16`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/34b4f168ddb4a170d37bfd0e1484e4060886a03f))

- Ensure isort compatible with black
  ([`dba11d2`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/dba11d279223027c0eb9f35c9c3369ee09446417))

- Update requirements.txt
  ([`75709f4`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/75709f40d9116948a62b2f24e6a8f5c942e2df79))

### Continuous Integration

- Update typecheck job to auto-install missing type stubs
  ([`ca89c5c`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/ca89c5cae8f3150cc48a641914ea44e971869b3e))

### Features

- Add argparsing for config path and enable tensor arguments to loss_fn
  ([`63b5e34`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/63b5e34a142cbe4ae538b623a19e7faa559e3698))

- Add lead identification
  ([`f77c9ec`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/f77c9ec9ae338d8b8090ec87a26cdf6ec8e572aa))

- Add signal extraction class
  ([`9d3a49f`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/9d3a49feeed6198ed36fa4e403a3985684040b54))

- Support lead identification
  ([`484410c`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/484410c030c246a92027de0e55cc1a8d95c59263))

- Update config files with for lead detection support
  ([`c000e39`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/c000e39ac0728fcbc57388f6f48d65bd7a349584))

### Refactoring

- Remove unused snake class
  ([`47286cf`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/47286cf71f5f323254671a9315f1955cdfdd6579))


## v1.4.0 (2025-06-25)

### Bug Fixes

- Convert apply_digital_redaction notebook to python script
  ([`1e28edf`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/1e28edfc78137c4189e4054af7a6648b59139e41))

- Visualize redaction percentage with plotly
  ([`34f3f6e`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/34f3f6e66dd8f507beee08f8cd2dcf41b625847e))

### Features

- Apply digital redaction to detected regions
  ([`3e7eaad`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/3e7eaad5f03bbbc3cde10e0fa43cc4de4847bf6c))

- Ensure images are properly redacted
  ([`f848077`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/f848077d907ac5ec0bc46093590724ec5dae9b90))

- Final parameters for redaction
  ([`48d487e`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/48d487e6c3fc956910e9122aed652ebcd9c88ef4))

- Implement adaptive redacted area finder
  ([`87a3eec`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/87a3eec3a1c14ad26545ffcb4cd495acd8283f15))


## v1.3.0 (2025-06-25)

### Bug Fixes

- Compile model instead of train fn, reduce verbosity, sort imports
  ([`8212841`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/821284154db3ab5e3c1cf8efe66953f447ca2ebf))

- Update requirements.txt
  ([`5724552`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/572455278e46e30e9ec8ae3cd8ff64e679d5c098))

### Chores

- Ignore sandbox dir for temporary output
  ([`577c0f1`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/577c0f1007a89130cee822536d418ff473cbea4e))

- Silence flake8 and mypy warnings
  ([`8c14d3a`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/8c14d3ae26da2a9bfd518a4acb663d211c9715ff))

- Update to python 3.12
  ([`5e2aa75`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/5e2aa7534b4d02752f0461e93da4689fa95d2cef))

### Code Style

- Sort imports
  ([`ff5b818`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/ff5b81862060590cdd52998b127bf22b3bef8ecd))

### Continuous Integration

- Add import sorting workflow
  ([`80b2c85`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/80b2c858296b6593896b0e1fe96325a94ae5b4a4))

- Do not use docker container for semantic-release
  ([`0ebc8c6`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/0ebc8c6e3bb224481fefe2b8ced35db66f535672))

- Use python 3.12
  ([`54fc6f3`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/54fc6f3141513033a060f902304979744eaeb342))

### Documentation

- Add python3.12 badge
  ([`2be6860`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/2be6860ebe65331bbc02219d3026d987ff366a40))

### Features

- Add Muon optimizer
  ([`26bba3e`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/26bba3e4c34b3279817ddd0975b3bc18de22168a))

- Add skip connections and learnable upsampling to unet
  ([`2ab8bc2`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/2ab8bc2841dd3e9bd2f3e4f73c1af4e7cae626c5))

- Better centering in cropper
  ([`2ae58e7`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/2ae58e7aec7a9133ea7cc26b78c9c7cbd39dbb77))

- Digitize hierarchical file structure
  ([`3ddc885`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/3ddc8856898dc3160d38c9ac1b0718e002866434))

- More robust autodetection of number of leads
  ([`528f36f`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/528f36fbfd0727d354ecb02808ee29fc4c4f8b06))

- Upgrade from 3 to 4 classes in segmentation model
  ([`54c4ba2`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/54c4ba2c3656dd0be2cfaf05034aa908c6496f7b))

- Visualization of feature map for use in training
  ([`aa47bd1`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/aa47bd1a58bd59dda48adb45d32909727ee8c67c))

### Refactoring

- Remove obsolete huggingface segmentation model
  ([`77768e6`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/77768e6f2a93631f72561c78f523760bac27936b))


## v1.2.0 (2025-01-21)

### Bug Fixes

- Fill average rgb image color for perspective
  ([`bb536be`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/bb536befc643f72f9754e1f594da436e86709e07))

### Continuous Integration

- Add ~/.local/bin to path
  ([`4805eb1`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/4805eb155bfc7e45310059383ec308ae8dc4b752))

- Use python3.10
  ([`1c7e999`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/1c7e999b3d8651402147102dad7eb7b7ce09d104))

- Use self-hosted runners as github has limit
  ([`7097921`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/70979216ee6b1687f652b4fa3638d21bb4c75f97))

### Features

- Add pixel_size_finder to InferenceWrapper
  ([`ec07675`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/ec07675f54c206a91b217e78e7ea01f30fe6cad7))

- Create a class to find mm/pixel in x and y direction
  ([`741bd5f`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/741bd5fe06ceec0e92532218209abec7be40aafa))


## v1.1.0 (2025-01-21)

### Features

- Separate cropping module
  ([`83edca6`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/83edca601ec0c749fd4da30d001127b7d4dec517))

### Refactoring

- Remove cropping logic from snake
  ([`bcdcfe6`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/bcdcfe61df68ad7c65e2ad6e9937d31a51c92f99))

- Remove grid detector class
  ([`2667d2d`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/2667d2d01cdaa3630c706072f2edd5f46fdec684))


## v1.0.0 (2025-01-17)

### Bug Fixes

- Raise error on empty dataloader
  ([`7fb0792`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/7fb0792cef5430b516eafe32acae81270aac8f03))

### Features

- Support png masks instead of npy masks
  ([`548f991`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/548f99163ecd937a96ba744cfa1d0453ee1a22ce))

### Refactoring

- Scans and masks are in same folder
  ([`e4be434`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/e4be434e5062eb7b67752d5056a570de14d1eb51))

### Breaking Changes

- Bump the dataset to have png masks instead of npy masks


## v0.15.0 (2025-01-16)

### Features

- Create initial hyp search plan for unet
  ([`c18e3cb`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/c18e3cbce4406ce1a897ccca89821fefe773b4c6))

- Create WeightedDiceLossSquared
  ([`7692b81`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/7692b813472fbd96a14e9231d9e106c3c363b6de))


## v0.14.3 (2025-01-15)

### Bug Fixes

- Use individual states for each run in EarlyStopper
  ([`603ba36`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/603ba36724db78cc60adc2231b4fa5c013f0acaa))


## v0.14.2 (2025-01-15)

### Bug Fixes

- Resize if background image is lower resolution than ecg
  ([`eb5967b`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/eb5967bc017540c85478abd7c34c82835b68664a))


## v0.14.1 (2025-01-14)

### Bug Fixes

- Do not train on validation set
  ([`6c634f0`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/6c634f0e4deba4d8a80b6082a57cbf61e5c2a1be))

- Start calculating metrics
  ([`88728ea`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/88728ea59056132efcf989b1bcbfa4260c9f59da))


## v0.14.0 (2025-01-13)

### Bug Fixes

- Load weights to correct device and do not sort snake
  ([`e85d518`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/e85d5181cc436de59a8886c385c0f938733f188c))

- Remove erosion and edge effects
  ([`33610fc`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/33610fca3c3fa04eeab0fa01098f4c6a2081d463))

- Specify RGB to avoid bug
  ([`d811287`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/d811287690ab231bd092e536c4801c3020ee6dc2))

### Features

- Optional automatic estimation of num peaks
  ([`9f9deff`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/9f9deff0b95abcbc86669cd82a063de8f74d0117))


## v0.13.0 (2025-01-13)

### Bug Fixes

- Configure hyperparameter search with searchspace kwargs
  ([`8c27bd1`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/8c27bd15285b09a06d4af82c51482fffa4bb1b33))

- Correct WeightedDiceLoss for batch size > 1
  ([`2f94966`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/2f94966935ac1b113e0fe21e9b3f3c173a70b517))

- Enforce uniform depth for all unet en-/decoder blocks
  ([`d2ab89e`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/d2ab89ea0edcba5eb6255f957852d447640aec65))

- Flip metric comparison sign as we only use loss
  ([`09b563d`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/09b563db57b24d9b7acc7c4b16cf5027ccf33c4d))

- Improve control flow
  ([`1591018`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/15910184133ced1634d1844300a1dbc9dd2ea8d9))

- Improve naming of in- & out channels for unet
  ([`44c3588`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/44c358854fe648a8933bd9d662e7ba611964f908))

- Make hyperparameter search scheduler optional
  ([`10711a1`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/10711a1386b592b9acd18355993981c10ee856b6))

- Seed hyperparameter search for uniform configuration each run
  ([`14b0a46`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/14b0a462710b280ad10a8119371f8f29e99debaa))

### Features

- Add learning rate scheduler
  ([`e3bf78b`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/e3bf78bfe8484b5fba1605568824cd6f91fb645c))

- Config file for unet hyperparameter search
  ([`c2236a6`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/c2236a682ea01ac6713919666b0c74471d05f10a))

- Enable custom union exponent
  ([`b1742e3`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/b1742e37abe36935331e4fe2149af2918a2c5da7))

- Enable early stopping of training
  ([`2eea233`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/2eea23322d442a20854e8493c4a6deab3d1b9868))

- Implement cosine to constant lr scheduler
  ([`12c950d`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/12c950db16c83dc4ce6bab12a4d4f8d6f0c5875b))

- Implement MulticlassBinaryDiceLoss
  ([`d25e3d5`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/d25e3d589655c3c72a665fdc38b9a38df346f48e))

- Support kwargs for metrics
  ([`5b1a4fe`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/5b1a4fe039d09c44ba10a8f960151d0cc9960b4e))

### Performance Improvements

- Calculate running metrics due to memory limitations
  ([`75609c7`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/75609c7c44b75cf765f7d2e0eeefd6049f396447))

- Configure unet hyperparameter search
  ([`845ece9`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/845ece93ccdb404d8232d286a2b5d2fbac482aa2))

- Increase performance by pinning dataloader memory
  ([`6da1eca`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/6da1eca25222a68f011897d5d7b8e9060680ab2b))

- Load data samples lazily due to memory constraints
  ([`7bd647a`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/7bd647a8b8706b53913e2b5433ccc7a0ac8092c8))

### Refactoring

- Create class for binarization of multiclass loss
  ([`e1e1625`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/e1e162553478ac56a19d2d35894777b4edc76d6b))

### Testing

- Remove compilation as it fails and is currently not necessary
  ([`f7fb902`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/f7fb9027973329eb02b702af37dad40c841c433a))


## v0.12.0 (2025-01-10)

### Bug Fixes

- Do not crash on bad segmentation input
  ([`827f10b`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/827f10bc580251508457d157b557caf11bf02204))

### Features

- Add maximum limit on nonzero pixels
  ([`7c2daa7`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/7c2daa7e9c8cbea925a1d2b1cefc8f6558972214))

- Digitization wrapper for inference
  ([`6be01c8`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/6be01c8116ee83e941b8a60f6c2a9ba13a281480))


## v0.11.0 (2025-01-08)

### Features

- Better snake init and interpolation of missing values
  ([`3979227`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/397922770e6fcac6712ab0c60eded8c405e1fc46))


## v0.10.0 (2025-01-08)

### Features

- Enable huggingface segmentation models
  ([`afc4b9a`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/afc4b9a9d5be8cbb032bef174f9a546dfb247541))


## v0.9.0 (2025-01-08)

### Features

- Cache transformed images to disc
  ([`37f8716`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/37f8716c664714d5cf94c7d6683e74d23a9ddda3))

- Perspective and updated random resize
  ([`a0b48cc`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/a0b48cc9e14996cbce2c66c3d2eeb0a97a3b83cf))


## v0.8.1 (2025-01-06)

### Bug Fixes

- Reduce edge effects for robustness
  ([`96b69d1`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/96b69d1bfb89f5c22adc2bf31e2bb1bfd2caabe6))

- Require grid pixels to be minima
  ([`1379dbe`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/1379dbea386b53d5c48d0621058569b5ad3f9dda))

### Documentation

- Example of perspective estimation on real image
  ([`b5b984d`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/b5b984d91c0e142c7f154bd6748e766783832b11))

- Illustrate perspective correction
  ([`098c0c9`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/098c0c97083b1522fc016930babb8fb819055650))

- Pgf figure image to angle/radius
  ([`f8bb721`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/f8bb721079d00b64b7ce08d780bd04f304baecdb))


## v0.8.0 (2024-12-27)

### Features

- Perspective detector handles binarization
  ([`2b743d2`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/2b743d28a4d2a85a8befb3885ebb9d1bbea59951))


## v0.7.0 (2024-12-27)

### Bug Fixes

- Probabilities sum to 1
  ([`8b477fe`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/8b477fe86cf9bf6fd9a73fead9377d566d645997))

### Features

- Add support for composed transforms
  ([`5b931bb`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/5b931bbacbbb52a1f649eafa44e1edd9bed48f22))

- Add transforms and optional reduced load size
  ([`01d2a59`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/01d2a59b2c777297e13e52b8cff377c222305765))


## v0.6.0 (2024-12-23)

### Features

- Perspective detection for ecg paper
  ([`e99d2ad`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/e99d2ad63237908593472003d12fde5aa8395a00))

### Testing

- Perspective detector returns reasonable src points
  ([`49e4c32`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/49e4c32722fd957cee6400c783693f02f067c1ba))


## v0.5.0 (2024-12-20)

### Features

- Ecg scan transforms and corresponding visualization
  ([`94192fd`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/94192fd83795aba32895472249c2634bb1d5ee28))


## v0.4.0 (2024-12-19)

### Bug Fixes

- Cast metrics output to numpy as tune does not support tensors
  ([`15c9819`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/15c98198ab12039c745e003858f1620c793b9a87))

- Correct input order of predictions and targets to metrics
  ([`bf62166`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/bf6216620612e5e9101660e5d81e1c01e8a934a7))

- Correct scaling of WeightedCrossEntropyLoss
  ([`2b97ab1`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/2b97ab172b2b7b9fec410d53898511a67ce45c85))

- Initialize metric classes
  ([`da3dc7e`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/da3dc7e69f0a1b5f24764e8631f73c4b6f814566))

- Patch off by one epoch offset
  ([`a8e9713`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/a8e971320c909de327ad2b9ef7b128b042563b77))

- Store raw torch tensors to from predictions and targets
  ([`a44930e`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/a44930e933ee0493b7d5d8f0869c4a28fa969b4f))

### Continuous Integration

- Increase max test time to 5 minutes
  ([`1611a5b`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/1611a5be6f7e73fa6c2b75fda6934f85971727f7))

### Documentation

- Add marker for google docstring convention
  ([`f258d28`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/f258d28d0a7577268122d3f5a264b571f3e002f6))

### Features

- Implement MulticlassBinaryCrossEntropyLoss
  ([`30b7aea`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/30b7aea13105685609822af41a5448b5cfc8a5d6))

- Set up tensorboard
  ([`ecea315`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/ecea3158821dc1cb45118a94017a7b1ac1fcf4bf))

- Split ecg dataset into train, val and test
  ([`6715f97`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/6715f977a16fe15afbd01df984eaf9973bfe6ed2))

- Train dummy segmentation network
  ([`5cb3264`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/5cb32645990dc9b4cf0a367c3a7f64211b2847d6))

### Performance Improvements

- Add flag for cudnn_benchmark
  ([`dc50b21`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/dc50b21734a3c0e697bcb086281ba11931d7ba8f))

- Compile train function if not using ray and mixed precision
  ([`0d7eb8d`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/0d7eb8dac14c95c03764011de9f8e8cde0bb3cf3))

- Support mixed precision
  ([`7f10101`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/7f10101a3239a5d7237ca92afd4754bca74da6dd))


## v0.3.0 (2024-12-18)

### Bug Fixes

- Remove superfluous argument
  ([`398d925`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/398d9254ed25722cc4f34321256f30c31f552c78))

- Remove torch.compile from inside model class
  ([`23ec2fe`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/23ec2febe7d51d55a7c3c727e03e2a1d0c64ea68))

- Rename to enable __call__
  ([`7047811`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/70478110c09a85dfc90d7a9630454ee02d576e2c))

### Documentation

- Add docstrings to grid detectors
  ([`097b6f4`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/097b6f4f4c36fd344036d1a884f50a4f291340d4))

### Features

- Multi-scale grid detection
  ([`7328f4e`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/7328f4edb59d0acfce8f82c40198643e5206ebd8))

### Testing

- Make sure multiscale grid detector runs
  ([`ed3e9d0`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/ed3e9d0f1afaee14cb9251cf8e5ceeaff053bc14))


## v0.2.0 (2024-12-16)

### Bug Fixes

- One hot encode target mask
  ([`f7592b9`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/f7592b96918105929d8d64b873889e759b7d60e7))

- Path for dice loss
  ([`3b7c3ba`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/3b7c3ba959ef4cee1d87f540263083c5676e7f60))

- Use custom loss
  ([`ccb697b`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/ccb697b8069d53007de44c44cbfdb211ba82fce5))

### Continuous Integration

- Ignore long comments
  ([`32d0c02`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/32d0c023e25ab801f9f6d0e1e43fd7ba37ca4b3e))

### Features

- Grid detection with cuda support
  ([`456f152`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/456f152b6a848ccbbaefdfc7e18e3f170c55e704))

- Snake and segmentation losses
  ([`7be3a0a`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/7be3a0a2ca9a6d55f2fcc7c827601fe8d3d0c182))

- Snake fitter, no cuda support yet
  ([`a0d659b`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/a0d659bb7ba0853d3f06d292392ecd5d6dab4848))

### Testing

- Change model params for faster testing
  ([`888ab20`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/888ab20cd1056830456d98e0014e98dbd70cb070))


## v0.1.0 (2024-12-13)

### Continuous Integration

- Add .gitignore
  ([`b9a29b2`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/b9a29b2386f63cee7507de64fa890e782186087f))

- Ignore E203 for flake8 as black handles formatting
  ([`40659ee`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/40659eec5abcda2fb19761d750d3fb367200d54d))

- Increase flake8 line length to 120
  ([`d0c522a`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/d0c522a4c00296fe6af49ed5d1adc17e9733c009))

- Typecheck python with mypy
  ([`70c6fdf`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/70c6fdf2b73fb5c1dd8edc1d46a261d070bfec3f))

- Use black in addition to flake8 for formatting
  ([`e3208b5`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/e3208b51fe37ee2269a370c4ea12f15d3060514c))

### Documentation

- Add test status badge
  ([`ddd700e`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/ddd700ea9cc0e3b65d949398070df6ce6a8b35cd))

### Features

- Set up project structure
  ([`a84603f`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/a84603f2f6745e01c4eb9171f782a2c7aa31786b))

### Testing

- Set up pytest
  ([`2641fe3`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/2641fe32ceb1c6802d0782ba6d1f80828d32e7e7))


## v0.0.0 (2024-12-06)

- Initial Release
