(function () {
  var adminEventSource = null;
  var adminEventsUrl = null;
  var adminStatusUrl = null;
  var adminRefreshPending = false;

  function applySwap(target, html, swap) {
    if (!target) {
      return;
    }
    if (swap === "outerHTML") {
      target.outerHTML = html;
      return;
    }
    target.innerHTML = html;
  }

  function findTarget(selector) {
    if (!selector) {
      return null;
    }
    return document.querySelector(selector);
  }

  function scrollMessages() {
    var stream = document.querySelector("[data-message-stream]");
    if (stream) {
      stream.scrollTop = stream.scrollHeight;
    }
  }

  function closeCitationPanel() {
    document.body.dataset.citationOpen = "false";
  }

  function openCitationPanel() {
    document.body.dataset.citationOpen = "true";
  }

  function scrollViewportToTop() {
    var detailCard = document.querySelector(".detail-card");
    if (detailCard) {
      detailCard.scrollTop = 0;
    }
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  function closeAdminEventSource() {
    if (adminEventSource) {
      adminEventSource.close();
      adminEventSource = null;
    }
    adminEventsUrl = null;
    adminStatusUrl = null;
    adminRefreshPending = false;
  }

  function setAdminStatusNotice(message) {
    var region = document.querySelector("#admin-run-region");
    if (!region) {
      return;
    }
    var errorNode = region.querySelector(".inline-error");
    if (errorNode) {
      errorNode.textContent = message;
      return;
    }
    var card = region.querySelector(".admin-status-card");
    if (!card) {
      return;
    }
    var notice = document.createElement("p");
    notice.className = "inline-error";
    notice.textContent = message;
    card.appendChild(notice);
  }

  function refreshAdminRunRegion() {
    if (adminRefreshPending || !adminStatusUrl) {
      return;
    }
    adminRefreshPending = true;
    fetch(adminStatusUrl, {
      method: "GET",
      credentials: "same-origin",
      headers: { "HX-Request": "true" }
    })
      .then(function (response) {
        if (response.redirected) {
          window.location.assign(response.url);
          return null;
        }
        return response.text();
      })
      .then(function (html) {
        if (!html) {
          return;
        }
        var region = document.querySelector("#admin-run-region");
        applySwap(region, html, "outerHTML");
        syncAdminRunRegion();
      })
      .catch(function () {
        closeAdminEventSource();
        setAdminStatusNotice("进度连接已中断或登录已失效，请刷新后重新登录。");
      })
      .finally(function () {
        adminRefreshPending = false;
      });
  }

  function syncAdminRunRegion() {
    var region = document.querySelector("#admin-run-region");
    if (!region) {
      closeAdminEventSource();
      return;
    }

    var nextEventsUrl = region.dataset.eventsUrl || null;
    var nextStatusUrl = region.dataset.statusUrl || null;
    var isTerminal = region.dataset.terminal === "true";

    if (!nextEventsUrl || !nextStatusUrl || isTerminal) {
      closeAdminEventSource();
      return;
    }

    if (adminEventSource && adminEventsUrl === nextEventsUrl && adminStatusUrl === nextStatusUrl) {
      return;
    }

    closeAdminEventSource();
    adminEventsUrl = nextEventsUrl;
    adminStatusUrl = nextStatusUrl;
    adminEventSource = new EventSource(adminEventsUrl);
    adminEventSource.addEventListener("progress", function () {
      refreshAdminRunRegion();
    });
    adminEventSource.addEventListener("error", function () {
      closeAdminEventSource();
      setAdminStatusNotice("进度连接已中断或登录已失效，请刷新后重新登录。");
    });
  }

  function handleSwap(target) {
    if (!target) {
      return;
    }
    if (target.id === "app-shell") {
      scrollMessages();
      closeCitationPanel();
    }
    if (target.id === "citation-detail") {
      openCitationPanel();
      scrollViewportToTop();
    }
    if (target.id === "admin-app-shell" || target.id === "admin-run-region") {
      syncAdminRunRegion();
    }
  }

  function performRequest(url, options, targetSelector, swap) {
    var target = findTarget(targetSelector);
    if (!target) {
      window.location.assign(url);
      return;
    }

    fetch(url, {
      method: options.method || "GET",
      body: options.body || null,
      credentials: "same-origin",
      headers: Object.assign({ "HX-Request": "true" }, options.headers || {})
    })
      .then(function (response) {
        if (response.redirected) {
          window.location.assign(response.url);
          return null;
        }
        return response.text().then(function (html) {
          return { response: response, html: html };
        });
      })
      .then(function (result) {
        if (!result) {
          return;
        }
        applySwap(target, result.html, swap || "innerHTML");
        handleSwap(findTarget(targetSelector));
      })
      .catch(function () {
        window.location.assign(url);
      })
      .finally(function () {
        if (options.pendingElement) {
          setPending(options.pendingElement, false);
        }
      });
  }

  function setPending(element, isPending) {
    if (!element) {
      return;
    }

    element.classList.toggle("is-requesting", isPending);
    element.classList.toggle("is-loading", isPending);
    element.setAttribute("aria-busy", isPending ? "true" : "false");

    var controls = element.querySelectorAll("button, input, textarea, select");
    controls.forEach(function (control) {
      control.disabled = isPending;
    });

    var submit = element.querySelector("button[type='submit']");
    if (!submit) {
      return;
    }
    if (isPending) {
      if (!submit.dataset.originalText) {
        submit.dataset.originalText = submit.textContent;
      }
      if (submit.closest("#ask-form")) {
        submit.textContent = "正在生成...";
      } else if (submit.closest(".admin-ingest-form")) {
        submit.textContent = "正在提交...";
      } else {
        submit.textContent = "处理中...";
      }
    } else if (submit.dataset.originalText) {
      submit.textContent = submit.dataset.originalText;
    }
  }

  document.addEventListener("DOMContentLoaded", function () {
    scrollMessages();
    syncAdminRunRegion();
  });

  document.body.addEventListener("click", function (event) {
    var remote = event.target.closest("[hx-post], [hx-get], [hx-delete]");
    if (event.target.closest("[data-close-citation]")) {
      closeCitationPanel();
      return;
    }

    if (!remote) {
      return;
    }

    if (remote.closest("form") && remote.type === "submit") {
      return;
    }
    if (remote.tagName === "FORM" && event.target !== remote) {
      return;
    }

    event.preventDefault();
    var hxPost = remote.getAttribute("hx-post");
    var hxGet = remote.getAttribute("hx-get");
    var hxDelete = remote.getAttribute("hx-delete");
    var url = hxPost || hxGet || hxDelete;
    if (!url) {
      return;
    }

    performRequest(
      url,
      { method: hxPost ? "POST" : hxDelete ? "DELETE" : "GET" },
      remote.getAttribute("hx-target"),
      remote.getAttribute("hx-swap")
    );
  });

  document.body.addEventListener("submit", function (event) {
    var form = event.target.closest("form[hx-post]");
    if (!form) {
      return;
    }

    event.preventDefault();
    var formData = new FormData(form);
    setPending(form, true);
    performRequest(
      form.getAttribute("hx-post"),
      {
        method: "POST",
        body: formData,
        pendingElement: form
      },
      form.getAttribute("hx-target"),
      form.getAttribute("hx-swap")
    );
  });
})();
