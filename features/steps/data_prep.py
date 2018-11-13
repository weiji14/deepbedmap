from behave import given, when, then
import os


@given(u"this {url} link to a file hosted on the web")
def set_url(context, url):
    context.url = url


@when(u"we download it to {filepath}")
def download_from_url_to_path(context, filepath):
    context.filepath = filepath
    context.data_prep.download_to_path(path=filepath, url=context.url)


@then(u"the local file should have this {sha256} checksum")
def check_sha256_of_file(context, sha256):
    assert context.data_prep.check_sha256(path=context.filepath) == sha256
    os.remove(path=context.filepath)  # remove downloaded file
