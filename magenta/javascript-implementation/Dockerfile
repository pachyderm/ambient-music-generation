FROM buildkite/puppeteer

RUN mkdir /samples
RUN mkdir /outputs

COPY src /code/src
COPY yarn.lock /code/yarn.lock
COPY package.json /code/package.json

WORKDIR /code

RUN yarn

CMD yarn transcribe
