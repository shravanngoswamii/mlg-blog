# MLG Blog

This is the repository for the MLG blog. It is built using Jekyll and hosted on GitHub Pages.

The blog is available at [https://mlg.eng.cam.ac.uk/blog](https://mlg.eng.cam.ac.uk/blog).

## Installation: Prerequisites

### Mac (Apple Silicon)

There is a bug in the current version of Jekyll that prevents it from using the latest version of Ruby. To get around this, you need to install an older version of Ruby using Homebrew.

```bash
brew install ruby@3.1
```

In your `~/.zshrc` file, add the following line (pointing to the location of the Ruby binary installed by Homebrew):

```bash
export PATH="/opt/homebrew/opt/ruby@3.1/bin:$PATH"
```

Don't forget to source your `~/.zshrc` file.

Test that you have the correct version of Ruby installed:

```bash
ruby -v
```
should output something like the above: `ruby 3.1.4p223 (2023-03-30 revision 957bb7cb81) [arm64-darwin21]`

## Installation

Prerequisites: Ruby (see above).

After cloning the repository, `cd` into the directory and install the dependencies following the instructions below.

Once the correct version of Ruby is installed, you can install Jekyll and Bundler. We need to specify the version of Bundler to install, otherwise it will install the latest version, which is incompatible with the version of Bundler that was used to build the lock file.
```bash
gem install jekyll bundler:2.2.15
```

Install the dependencies:

```bash
bundle install
```

Finally, run the server in development mode:

```bash
bundle exec jekyll serve --drafts
```
