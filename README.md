
Tools and dependencies:

- [Chirpy](https://github.com/cotes2020/jekyll-theme-chirpy) Jekyll Theme
- [Jekyll Compose](https://github.com/jekyll/jekyll-compose)

Running server (http://localhost:4000):

```shell
./script/start
```

Creating a new draft:

```shell
bundle exec jekyll compose "My new draft" --draft
```

Publishing a draft:

```shell
bundle exec jekyll publish _drafts/my-new-draft.md
```

Renaming or changing the date of a post:

```shell
bundle exec jekyll rename _posts/2014-01-24-my-new-post.md "My Old Post" --date "2012-03-04"
```
