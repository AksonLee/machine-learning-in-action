use ansi_term::Color;
// use ansi_term::Style;

fn main() {
    println!("{}", Color::Red.blink().paint("Have a nice day!"));
}
