package com.sih26168.drnav

import android.os.Build
import org.json.JSONArray
import org.json.JSONObject
import java.io.BufferedWriter
import java.io.File
import java.io.FileWriter

/**
 * Writes a recording directory in exactly the layout `drnav.log_format`
 * expects, so a drive logged here replays offline with no conversion:
 *
 * ```
 * <recording>/ meta.json  imu.csv  gnss.csv  gnss_raw.csv  baro.csv
 * ```
 *
 * CSV is a deliberate choice over a binary format or a database. It survives
 * being emailed round the team, opens in Excel when someone wants to eyeball a
 * column, and needs no parsing library on either side.
 *
 * Writes are buffered and flushed on [close]; at 200 Hz across six channels
 * this is roughly 4 MB per minute of recording, which is fine for the drive
 * lengths involved but worth remembering before logging for an hour.
 */
class LogWriter(private val dir: File) {

    private val imu: BufferedWriter
    private val gnss: BufferedWriter
    private val gnssRaw: BufferedWriter
    private val baro: BufferedWriter

    var imuRows = 0L; private set
    var gnssRows = 0L; private set

    init {
        dir.mkdirs()
        imu = open("imu.csv", "t,ax,ay,az,gx,gy,gz")
        gnss = open("gnss.csv", "t,e,n,u,ve,vn,vu,nsat,cn0,hdop,acc")
        gnssRaw = open(
            "gnss_raw.csv",
            "t,svid,constellation,cn0,pseudorange_rate,pseudorange_rate_unc," +
                "carrier_freq_hz,multipath,state",
        )
        baro = open("baro.csv", "t,pressure_hpa")
    }

    private fun open(name: String, header: String): BufferedWriter =
        BufferedWriter(FileWriter(File(dir, name)), 1 shl 16).apply {
            write(header); newLine()
        }

    fun imu(t: Double, a: FloatArray, g: FloatArray) {
        imu.write("${f(t)},${f(a[0])},${f(a[1])},${f(a[2])},${f(g[0])},${f(g[1])},${f(g[2])}")
        imu.newLine()
        imuRows++
    }

    fun gnss(
        t: Double, e: Double, n: Double, u: Double,
        ve: Double, vn: Double, vu: Double,
        nsat: Int, cn0: Double, hdop: Double, acc: Double,
    ) {
        gnss.write("${f(t)},${f(e)},${f(n)},${f(u)},${f(ve)},${f(vn)},${f(vu)}," +
            "$nsat,${f(cn0)},${f(hdop)},${f(acc)}")
        gnss.newLine()
        gnssRows++
    }

    fun gnssRaw(
        t: Double, svid: Int, constellation: Int, cn0: Double,
        prRate: Double, prRateUnc: Double, carrierHz: Double,
        multipath: Int, state: Int,
    ) {
        gnssRaw.write("${f(t)},$svid,$constellation,${f(cn0)},${f(prRate)}," +
            "${f(prRateUnc)},${f(carrierHz)},$multipath,$state")
        gnssRaw.newLine()
    }

    fun baro(t: Double, hpa: Float) {
        baro.write("${f(t)},${f(hpa)}"); baro.newLine()
    }

    /** Writes `meta.json`; must be called once the origin fix is known. */
    fun writeMeta(
        mode: String, lat0: Double, lon0: Double, alt0: Double,
        imuRateHz: Double, notes: String, events: List<JSONObject> = emptyList(),
    ) {
        val meta = JSONObject().apply {
            put("mode", mode)
            put("source", "android")
            put("imu_rate_hz", imuRateHz)
            put("gnss_rate_hz", 1.0)
            put("lat0", lat0); put("lon0", lon0); put("alt0", alt0)
            put("device", "${Build.MANUFACTURER} ${Build.MODEL} (API ${Build.VERSION.SDK_INT})")
            put("notes", notes)
            put("mount", JSONObject.NULL)
            put("events", JSONArray(events))
        }
        File(dir, "meta.json").writeText(meta.toString(2))
    }

    fun close() {
        listOf(imu, gnss, gnssRaw, baro).forEach {
            runCatching { it.flush(); it.close() }
        }
    }

    /** Fixed 6 decimals: plenty for metres and rad/s, and keeps files compact. */
    private fun f(v: Number) = String.format(java.util.Locale.US, "%.6f", v.toDouble())
}
