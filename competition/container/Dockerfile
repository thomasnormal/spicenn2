# Fixed base manifest and upstream ngspice release commit. The final image ID
# must also be recorded by the organizer: Debian build dependencies can change.
FROM python:3.11-slim-bookworm@sha256:528257d48c1da0dcecc2e725d1ae34498d60c965f1241e39cd6a85a8859bdf84 AS build
RUN apt-get update && apt-get install --yes --no-install-recommends \
    autoconf automake bison build-essential ca-certificates flex git libtool
WORKDIR /build
RUN git init ngspice && cd ngspice && \
    git remote add origin https://git.code.sf.net/p/ngspice/ngspice && \
    git fetch --depth 1 origin ebdaf58ec76a06ffaac7e0f138360dd1cf5ee4b6 && \
    git checkout --detach FETCH_HEAD && \
    ./autogen.sh && \
    ./configure --prefix=/opt/ngspice --without-x --with-readline=no \
    --enable-xspice --enable-cider --disable-debug --disable-openmp && \
    make -j2 && make install
RUN git -C /build/ngspice archive --format=tar HEAD | gzip -n > /opt/ngspice-source.tar.gz

FROM python:3.11-slim-bookworm@sha256:528257d48c1da0dcecc2e725d1ae34498d60c965f1241e39cd6a85a8859bdf84
COPY --from=build /opt/ngspice /opt/ngspice
COPY --from=build /build/ngspice/COPYING /usr/share/doc/ngspice/COPYING
COPY --from=build /opt/ngspice-source.tar.gz /usr/share/doc/ngspice/source.tar.gz
COPY competition/container/Dockerfile /usr/share/doc/ngspice/Dockerfile
ENV PATH="/opt/ngspice/bin:${PATH}"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# This image is only the simulator. The trusted host runner supplies the harness
# and scores the trace, so Python code changes do not silently change its image ID.
COPY LICENSE /usr/share/doc/spicenn2/LICENSE
WORKDIR /work
USER 65534:65534
ENTRYPOINT ["/opt/ngspice/bin/ngspice"]
