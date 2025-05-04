import git


def get_commit_hash() -> str:
    repository = git.Repo(search_parent_directories=True)
    sha = repository.head.object.hexsha

    return sha
