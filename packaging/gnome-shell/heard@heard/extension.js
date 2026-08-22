// heard window tools — D-Bus bridge for the heard voice daemon.
//
// GNOME Shell restricts org.gnome.Shell.Eval to unsafe mode (41+), so this
// extension exposes the few window operations heard needs as a proper
// session-bus service:
//
//   bus:      dev.heard.WindowTools  (session)
//   path:     /dev/heard/WindowTools
//   interface dev.heard.WindowTools
//
//     ListWindows()    -> s   JSON [{id, cls, title}, ...]
//     Activate(id i)
//     Close(target s)         "active" or a listed id (as string)
//     ToggleFullscreen()
//     SetWorkspace(n i)       1-based, matches heard's workspace numbers
//
// Window ids follow MetaWindow.get_stable_sequence() where available and
// fall back to get_id() on older shells; heard treats them opaquely.

import Gio from 'gi://Gio';
import GLib from 'gi://GLib';

const IFACE_XML = `
<node>
  <interface name="dev.heard.WindowTools">
    <method name="ListWindows">
      <arg type="s" direction="out" name="json"/>
    </method>
    <method name="Activate">
      <arg type="i" direction="in" name="id"/>
    </method>
    <method name="Close">
      <arg type="s" direction="in" name="target"/>
    </method>
    <method name="ToggleFullscreen"/>
    <method name="SetWorkspace">
      <arg type="i" direction="in" name="n"/>
    </method>
  </interface>
</node>`;

const PATH = '/dev/heard/WindowTools';
const NAME = 'dev.heard.WindowTools';

function idOf(w) {
    if (typeof w.get_stable_sequence === 'function')
        return w.get_stable_sequence();
    return w.get_id();
}

function allWindows() {
    return global.get_window_actors().map(a => a.meta_window);
}

function fail(msg) {
    throw new GLib.Error(GLib.quark_from_string('dev.heard.WindowTools.Error'),
                         0, msg);
}

function findWindow(id) {
    const want = String(id);
    const w = allWindows().find(m => String(idOf(m)) === want);
    if (!w)
        fail(`no window with id ${want}`);
    return w;
}

const Actions = {
    ListWindows() {
        return JSON.stringify(allWindows().map(w => ({
            id: idOf(w),
            cls: w.get_wm_class() || '',
            title: w.get_title() || '',
        })));
    },

    Activate(id) {
        findWindow(id).activate(global.get_current_time());
    },

    Close(target) {
        const w = target === 'active'
            ? global.display.focus_window
            : findWindow(target);
        if (!w)
            fail('no focused window');
        w.delete(global.get_current_time());
    },

    ToggleFullscreen() {
        const w = global.display.focus_window;
        if (!w)
            fail('no focused window');
        if (w.fullscreen)
            w.unfullscreen();
        else
            w.fullscreen();
    },

    SetWorkspace(n) {
        const ws = global.workspace_manager.get_workspace_by_index(n - 1);
        if (!ws)
            fail(`no workspace ${n}`);
        ws.activate(global.get_current_time());
    },
};

export default class HeardWindowTools extends Extension {
    enable() {
        this._ownerId = null;
        const iface =
            Gio.DBusNodeInfo.new_for_xml(IFACE_XML).lookup_interface(NAME);
        this._ownerId = Gio.bus_own_name(Gio.BusType.SESSION, NAME,
            Gio.BusNameOwnerFlags.NONE,
            connection => {
                connection.register_object(PATH, iface,
                    (conn, _sender, _path, _iface, method, params, invocation) =>
                        this._dispatch(method, params, invocation));
            },
            null, null);
    }

    disable() {
        if (this._ownerId !== null) {
            Gio.bus_unown_name(this._ownerId);
            this._ownerId = null;
        }
    }

    _dispatch(method, params, invocation) {
        try {
            const args = params.deepUnpack ? params.deepUnpack() : [];
            const out = Actions[method](...args);
            if (out === undefined)
                invocation.return_value(new GLib.Variant('()', []));
            else
                invocation.return_value(new GLib.Variant('(s)', [out]));
        } catch (e) {
            invocation.return_dbus_error('dev.heard.WindowTools.Error',
                                         String(e && e.message ? e.message : e));
        }
    }
}
