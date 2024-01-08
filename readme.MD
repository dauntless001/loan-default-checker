# Equity Share

Equity Share tokenizes income-producing properties on the Ethereum blockchain and allows retail investors the opportunity to own fractional shares of the property’s value.

Buy, sell, and exchange various Real Estate Tokens backed by specific target assets.


## Useful links

- [GitHub repo](https://github.com/equitycoin/equityshare)

## Requirements

- Python >= 3.8 / Django 3.2

## Project setup

If not using docker, you can setup a virtual environment using the command below

```sh
python -m venv env
```

then activate it with

```sh
./env/Scripts/activate   # windows or
source env/bin/activate  # linux or mac
```

## Install required packages

Run the command below

```sh
python -m pip install -r requirements.txt
```

Once the virtual environment has been activated, install the necessary requirements by using the command below

```sh
python manage.py migrate
```

## Database backups and restore

Sometimes you may want to delete your database but don't want to lose your data. To do this, run the following script



## Important steps


### Dev complete

- Once coding is done, run `git commit -m "short description"`
- If successful, `git push` your code to Gitlab
- Create a merge request using the link generated from the terminal

### Code reviews

- Notify two colleagues to perform code reviews
- If code review is successful, merge to develop and move ticket to `closed` lane on Gitlab

## Contribution

Pick a ticket on the [Gitlab repository](https://github.com/equitycoin/equityshare). If you haven't cloned the repository, use the command to clone from the terminal

```sh
git clone https://github.com/equitycoin/equityshare
```

When creating a new branch, **ENSURE** that the branch name starts with the format **LOA-&lt;issue-no&gt;-&lt;short-description&gt;** e.g. **LOA-1-project-setup** and the main branch is from develop. use the command below when creating a new branch.

```
git checkout develop
git branch -b <branch name>
```

Before creating a pull request, run the commands and fix any warning/errors encountered

```
git add .
git commit -m "my commit message"
git push -u origin <branch name>
```

When creating a pull request, please select the target branch as `develop`.



## Pushing to the repository

Run the following command

```sh
 # if pushing for the first time
$ git push -u origin <branchname>

# if pushing normally
$ git push
```
